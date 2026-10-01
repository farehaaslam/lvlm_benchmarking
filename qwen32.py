import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoProcessor,
    DynamicCache,
    DynamicLayer,
    Qwen3VLForConditionalGeneration,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask
from transformers.video_utils import VideoMetadata

# Plain HF transformers (no vLLM). Speed comes from:
#   1. each video is decoded, vision-encoded and prefilled ONCE; every
#      question reuses that KV cache and only prefills its own tokens
#   2. questions of a video are decoded together in a batch on top of the
#      shared video prefix (preallocated cache, no torch.cat per step)
#   3. GQA attention over the padded batch without copying K/V 8x
#
#   uv run qwen32.py 2>&1 | tee run_hf.log
#   uv run qwen32.py --limit 1          # smoke test


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

VIDEO_EXTENSIONS = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"]

# Exactly NUM_FRAMES frames are sampled uniformly over the full video
# (same as qwen32B_vllm.py, so answers are comparable).
NUM_FRAMES = 256

MAX_NEW_TOKENS = 2048

QUESTION_COLUMN = "Questions"

# Identical to qwen32B_vllm.py.
PROMPT_TEMPLATE = (
    "Answer the following question "
    "using the information available "
    "in the video.\n\n"
    "Question: {question}\n\n"
)

# Questions decoded together on top of one video's KV cache.
# Each extra row costs one copy of the video KV cache (~0.26 MB/token),
# so this is halved automatically on OOM.
QUESTION_BATCH_SIZE = 16

# Inputs the model forward needs from the processor output.
MODEL_INPUT_KEYS = [
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values_videos",
    "video_grid_thw",
]


# ============================================================
# ATTENTION: GQA WITHOUT REPEATING K/V
# ============================================================

def shared_kv_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """SDPA for padded batches that does not expand K/V to all query heads.

    HF's sdpa path calls repeat_kv (an 8x copy of the whole KV cache, every
    layer, every step) as soon as there is a padding mask. Instead, fold the
    query heads that share a KV head into the query length dimension, so
    SDPA sees plain multi-head attention with num_key_value_heads heads.
    """

    groups = getattr(module, "num_key_value_groups", 1)

    # No padding (prefix prefill, vision encoder, equal-length batches):
    # HF's own path already uses enable_gqa / is_causal there.
    if attention_mask is None or groups == 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )

    batch, num_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # query head h uses kv head h // groups -> (batch, kv_heads, groups * q_len, dim)
    query = query.reshape(batch, num_kv_heads, groups * q_len, head_dim)

    mask = attention_mask.unsqueeze(2).expand(-1, -1, groups, -1, -1)
    mask = mask.reshape(batch, attention_mask.shape[1], groups * q_len, -1)

    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scaling,
    )

    out = out.reshape(batch, num_heads, q_len, head_dim)

    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_shared_kv", shared_kv_sdpa)
AttentionMaskInterface.register("sdpa_shared_kv", sdpa_mask)


# ============================================================
# KV CACHE: VIDEO PREFIX SHARED BY A BATCH OF QUESTIONS
# ============================================================

class PreallocLayer(DynamicLayer):
    """KV cache layer with a fixed-size buffer.

    DynamicLayer does torch.cat on every decode step, i.e. it copies the
    whole (batch x video length) cache each token. Here new tokens are
    written in place and the returned keys/values are views.
    """

    def __init__(self, keys, values, batch_size, capacity):

        super().__init__()

        _, num_heads, prefix_len, head_dim = keys.shape

        self.key_buf = keys.new_empty(batch_size, num_heads, capacity, head_dim)
        self.value_buf = values.new_empty(batch_size, num_heads, capacity, head_dim)

        self.key_buf[:, :, :prefix_len] = keys
        self.value_buf[:, :, :prefix_len] = values

        self.length = prefix_len
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

        self._sync_views()

    def _sync_views(self):

        self.keys = self.key_buf[:, :, : self.length]
        self.values = self.value_buf[:, :, : self.length]

    def update(self, key_states, value_states, *args, **kwargs):

        end = self.length + key_states.shape[-2]

        self.key_buf[:, :, self.length:end] = key_states
        self.value_buf[:, :, self.length:end] = value_states

        self.length = end
        self._sync_views()

        return self.keys, self.values

    def get_seq_length(self):

        return self.length

    def batch_select_indices(self, indices):

        indices = indices.to(self.device)

        self.key_buf = self.key_buf[indices]
        self.value_buf = self.value_buf[indices]

        self._sync_views()


def expand_prefix_cache(model, prefix_cache, batch_size, capacity):

    cache = DynamicCache(config=model.config)

    cache.layers = [
        PreallocLayer(layer.keys, layer.values, batch_size, capacity)
        for layer in prefix_cache.layers
    ]

    return cache


# ============================================================
# HELPERS
# ============================================================

def find_videos():

    videos = []

    for extension in VIDEO_EXTENSIONS:
        videos.extend(VIDEO_DIR.glob(extension))

    return sorted(videos, key=lambda x: x.name)


def load_output(question_file, output_file):

    questions_df = pd.read_csv(question_file)

    if QUESTION_COLUMN not in questions_df.columns:
        raise ValueError(
            f"Column '{QUESTION_COLUMN}' not found in {question_file}. "
            f"Available columns: {list(questions_df.columns)}"
        )

    fresh = pd.DataFrame({
        "question": questions_df[QUESTION_COLUMN].astype(str),
        "answer": "",
    })

    if not output_file.exists():
        return questions_df, fresh

    output_df = pd.read_csv(output_file, keep_default_na=False)

    if (
        "question" not in output_df.columns
        or "answer" not in output_df.columns
        or len(output_df) != len(questions_df)
    ):
        print("[WARNING] Existing output does not match the question CSV. Reinitializing.")
        return questions_df, fresh

    output_df = output_df[["question", "answer"]].copy()
    output_df["answer"] = output_df["answer"].astype(str)

    return questions_df, output_df


def ffmpeg_frames(video_path, indices, fps, width, height):
    """Grab single frames with the system ffmpeg (accurate input seeking).

    Fallback for videos decord cannot seek in (VP9 in MP4 fails with
    "threaded_decoder.cc:104: Check failed: run_.load()").
    """

    frame_bytes = width * height * 3

    def grab(index):

        out = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-threads", "2",
                "-ss", f"{index / fps:.6f}", "-i", str(video_path),
                "-map", "0:v:0", "-frames:v", "1",
                "-s", f"{width}x{height}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ],
            capture_output=True,
            check=True,
        ).stdout

        # Seeking past the last decodable frame returns nothing; step back.
        if len(out) < frame_bytes and index > 0:
            return grab(max(index - int(round(fps)), 0))

        if len(out) < frame_bytes:
            raise RuntimeError(f"ffmpeg returned no frame at index {index} of {video_path}")

        return np.frombuffer(out[:frame_bytes], dtype=np.uint8).reshape(height, width, 3)

    with ThreadPoolExecutor(max_workers=16) as pool:
        return np.stack(list(pool.map(grab, indices)))


def decode_video(video_path):
    """Decode NUM_FRAMES uniformly spaced frames once per video."""

    t0 = time.time()

    reader = VideoReader(str(video_path), ctx=cpu(0))

    total = len(reader)
    fps = float(reader.get_avg_fps())
    height, width = reader[0].shape[:2]

    indices = np.linspace(0, total - 1, min(NUM_FRAMES, total)).round().astype(int)

    backend = "decord"

    try:
        frames = reader.get_batch(indices.tolist()).asnumpy()
    except DECORDError as e:
        print(f"[WARNING] decord failed on {video_path.name} ({e!r:.80}); using ffmpeg")
        backend = "ffmpeg"
        frames = ffmpeg_frames(video_path, indices.tolist(), fps, width, height)

    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        width=frames.shape[2],
        height=frames.shape[1],
        duration=total / fps,
        video_backend=backend,
        frames_indices=indices.tolist(),
    )

    del reader

    return frames, metadata, time.time() - t0


def build_prompt(processor, question):

    # Video comes FIRST so every question of a video shares the same prefix.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": PROMPT_TEMPLATE.format(question=question)},
            ],
        }
    ]

    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def common_prefix_length(sequences):

    first = sequences[0]
    length = min(len(s) for s in sequences)

    for other in sequences[1:]:
        for i in range(length):
            if first[i] != other[i]:
                length = i
                break

    return length


# ============================================================
# VIDEO PREFIX (vision encoder + prefill, once per video)
# ============================================================

def prefill_video(model, processor, frames, metadata, questions):
    """Encode the video and prefill everything the questions share.

    Returns the prefix KV cache, its length, the M-RoPE offset for text
    after the video, and the question-specific token ids of each question.
    """

    tokenizer = processor.tokenizer

    prompts = [build_prompt(processor, q) for q in questions]

    # Unexpanded prompts (one video placeholder): only used to find where
    # the questions start to differ.
    raw_ids = [
        tokenizer(p, add_special_tokens=False)["input_ids"]
        for p in prompts
    ]

    raw_prefix_len = common_prefix_length(raw_ids)

    # At least one question token must remain to produce the first logits.
    raw_prefix_len = min(raw_prefix_len, min(len(ids) for ids in raw_ids) - 1)

    t0 = time.time()

    inputs = processor(
        text=[prompts[0]],
        videos=[frames],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors="pt",
    )

    t1 = time.time()

    full_ids = inputs["input_ids"][0].tolist()

    # The processor only expands the video placeholder, so everything after
    # the video is the same tokens shifted by the expansion length.
    prefix_len = raw_prefix_len + len(full_ids) - len(raw_ids[0])

    if full_ids[prefix_len:] != raw_ids[0][raw_prefix_len:]:
        raise RuntimeError("Expanded prompt does not line up with the raw prompt.")

    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    if video_token_id in full_ids[prefix_len:]:
        raise RuntimeError("Shared prefix ends inside the video tokens.")

    seq_len = inputs["input_ids"].shape[1]

    prefix_inputs = {}

    for key in MODEL_INPUT_KEYS:

        if key not in inputs:
            continue

        value = inputs[key]

        # Per-token tensors are cut to the prefix; visual tensors stay whole.
        if value.ndim == 2 and value.shape == (1, seq_len):
            value = value[:, :prefix_len]

        prefix_inputs[key] = value.to(model.device)

    cache = DynamicCache(config=model.config)

    model.model.rope_deltas = None

    outputs = model(
        **prefix_inputs,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t2 = time.time()

    # Text after the video sits at M-RoPE position (index + rope_delta).
    rope_delta = int(model.model.rope_deltas.reshape(-1)[0].item())

    suffixes = [ids[raw_prefix_len:] for ids in raw_ids]

    print(
        f"Processor: {t1 - t0:.2f}s | video prefill: {t2 - t1:.2f}s | "
        f"prompt tokens: {seq_len} (shared prefix {prefix_len})"
    )

    del outputs, inputs, prefix_inputs

    return cache, prefix_len, rope_delta, suffixes


# ============================================================
# BATCHED GREEDY DECODING ON TOP OF THE PREFIX
# ============================================================

def answer_batch(model, prefix_cache, prefix_len, rope_delta, suffixes, eos_ids, pad_id):

    device = model.device

    batch = len(suffixes)
    lengths = [len(s) for s in suffixes]
    max_len = max(lengths)

    capacity = prefix_len + max_len + MAX_NEW_TOKENS

    cache = expand_prefix_cache(model, prefix_cache, batch, capacity)

    # Layout of every row: [video prefix][left padding][question tokens][answer]
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    position_ids = torch.full((batch, max_len), prefix_len + rope_delta, dtype=torch.long)
    attention_mask = torch.ones((batch, capacity), dtype=torch.long)

    for row, suffix in enumerate(suffixes):

        pad = max_len - len(suffix)

        input_ids[row, pad:] = torch.tensor(suffix)
        position_ids[row, pad:] = torch.arange(len(suffix)) + prefix_len + rope_delta
        attention_mask[row, prefix_len:prefix_len + pad] = 0

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Text tokens use the same position on all three M-RoPE axes.
    position_ids = position_ids.to(device).unsqueeze(0).expand(3, -1, -1)

    next_positions = torch.tensor(lengths, device=device) + prefix_len + rope_delta

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask[:, : prefix_len + max_len],
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )

    next_tokens = outputs.logits[:, -1].argmax(-1).to(device)

    cache_len = prefix_len + max_len

    rows = list(range(batch))
    generated = [[] for _ in range(batch)]
    cut_off = [False] * batch

    while True:

        keep = []

        for i, (row, token) in enumerate(zip(rows, next_tokens.tolist())):

            if token in eos_ids:
                continue

            generated[row].append(token)

            if len(generated[row]) >= MAX_NEW_TOKENS:
                cut_off[row] = True
                continue

            keep.append(i)

        if not keep:
            break

        # Drop finished questions so they stop costing compute.
        if len(keep) < len(rows):

            index = torch.tensor(keep, device=device)

            cache.batch_select_indices(index)

            next_tokens = next_tokens[index]
            next_positions = next_positions[index]
            attention_mask = attention_mask[index]

            rows = [rows[i] for i in keep]

        outputs = model(
            input_ids=next_tokens[:, None],
            attention_mask=attention_mask[:, : cache_len + 1],
            position_ids=next_positions.view(1, -1, 1).expand(3, -1, -1),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

        next_tokens = outputs.logits[:, -1].argmax(-1).to(device)
        next_positions = next_positions + 1
        cache_len += 1

    del cache, outputs

    return generated, cut_off


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(model, processor, job, decoded, batch_size, eos_ids, pad_id):

    video_name = job["name"]
    output_df = job["output_df"]
    pending = job["pending"]

    frames, metadata, decode_time = decoded

    print("\n" + "=" * 70)
    print(f"VIDEO: {video_name} | {len(pending)} questions | "
          f"{len(frames)} frames decoded in {decode_time:.2f}s")
    print("=" * 70)

    t0 = time.time()

    questions = [output_df.loc[idx, "question"] for idx in pending]

    prefix_cache, prefix_len, rope_delta, suffixes = prefill_video(
        model, processor, frames, metadata, questions
    )

    del frames

    tokenizer = processor.tokenizer
    generated_total = 0
    t_decode = time.time()

    start = 0

    while start < len(pending):

        end = min(start + batch_size, len(pending))

        try:

            t_batch = time.time()

            generated, cut_off = answer_batch(
                model, prefix_cache, prefix_len, rope_delta,
                suffixes[start:end], eos_ids, pad_id,
            )

        except torch.OutOfMemoryError:

            torch.cuda.empty_cache()

            if batch_size == 1:
                raise

            batch_size //= 2
            print(f"[OOM] retrying with question batch size {batch_size}")
            continue

        for offset, (tokens, was_cut) in enumerate(zip(generated, cut_off)):

            idx = pending[start + offset]

            answer = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            output_df.loc[idx, "answer"] = answer

            print(f"\n[{video_name}] Q{idx + 1}: {output_df.loc[idx, 'question']}")
            print(f"A: {answer}")

            if was_cut:
                print(f"[WARNING] answer hit MAX_NEW_TOKENS={MAX_NEW_TOKENS} and was cut off")

        batch_tokens = sum(len(g) for g in generated)
        generated_total += batch_tokens

        print(
            f"\nBatch {start + 1}-{end}/{len(pending)}: {time.time() - t_batch:.2f}s | "
            f"{batch_tokens / max(time.time() - t_batch, 1e-6):.1f} tok/s"
        )

        # Save after every batch (resume support).
        output_df.to_csv(job["output_file"], index=False)

        start = end

    del prefix_cache

    torch.cuda.empty_cache()

    elapsed = time.time() - t0

    print(
        f"\nCompleted: {video_name} in {elapsed:.2f}s | "
        f"{generated_total / max(time.time() - t_decode, 1e-6):.1f} output tok/s | "
        f"output: {job['output_file']}"
    )

    return batch_size


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=QUESTION_BATCH_SIZE,
                        help="questions decoded together per video (halved on OOM)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N videos (smoke test)")
    return parser.parse_args()


def build_jobs(videos):

    jobs = []

    for video_path in videos:

        video_name = video_path.stem
        question_file = QUESTION_DIR / f"{video_name}.csv"
        output_file = OUTPUT_DIR / f"{video_name}_answers.csv"

        if not question_file.exists():
            print(f"[WARNING] Question file not found: {question_file}")
            continue

        try:
            _, output_df = load_output(question_file, output_file)
        except ValueError as e:
            print(f"[ERROR] {e}")
            continue

        pending = [
            idx for idx in range(len(output_df))
            if str(output_df.loc[idx, "answer"]).strip() == ""
        ]

        if not pending:
            print(f"[SKIP] {video_name}: all questions already answered")
            continue

        jobs.append({
            "video_path": video_path,
            "name": video_name,
            "output_df": output_df,
            "output_file": output_file,
            "pending": pending,
        })

    return jobs


def main():

    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    videos = find_videos()
    if args.limit:
        videos = videos[:args.limit]

    jobs = build_jobs(videos)

    print(f"Found {len(videos)} videos | to process: {len(jobs)} | "
          f"questions pending: {sum(len(j['pending']) for j in jobs)}")

    if not jobs:
        print("Nothing to do.")
        return

    print("=" * 70)
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | {props.total_memory / 1024**3:.2f} GB")
    print("=" * 70)

    # Decode upcoming videos on CPU while the GPUs work.
    decoder = ThreadPoolExecutor(max_workers=2)
    futures = {0: decoder.submit(decode_video, jobs[0]["video_path"])}

    print(f"\nLoading {args.model}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa_shared_kv",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model)

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = next(iter(eos_ids))

    batch_size = args.batch_size
    t_start = time.time()

    with torch.inference_mode():

        for i, job in enumerate(jobs):

            if i + 1 < len(jobs):
                futures[i + 1] = decoder.submit(decode_video, jobs[i + 1]["video_path"])

            try:
                decoded = futures.pop(i).result()
                batch_size = process_video(
                    model, processor, job, decoded, batch_size, eos_ids, pad_id
                )
            except Exception as e:
                print(f"\n[ERROR] Video: {job['name']}\nError: {repr(e)}")
                job["output_df"].to_csv(job["output_file"], index=False)
                torch.cuda.empty_cache()

    decoder.shutdown()

    print("\n" + "=" * 70)
    print(f"ALL VIDEOS COMPLETED in {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
