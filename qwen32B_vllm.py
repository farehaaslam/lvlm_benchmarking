import os

# Must be set before vLLM is imported: worker processes are spawned (not
# forked), so the background video-decode threads below are safe.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    # vLLM >= 0.2x: uniform-sampling loader ("opencv")
    from vllm.multimodal.video import VideoBackend as UniformVideoLoader
except ImportError:
    # older vLLM (0.11 - 0.1x)
    from vllm.multimodal.video import OpenCVVideoBackend as UniformVideoLoader

# rclone copy gdrive:gurrt/gurrt_dataset/videos /workspace/lvlm_benchmarking/dataset/videos
# to run this script (uses every visible GPU, tensor parallel auto-picked):
#   uv run qwen32B_vllm.py 2>&1 | tee run.log
# quick smoke test on the first 2 videos:
#   uv run qwen32B_vllm.py --limit 2


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DATASET_DIR = Path("dataset").resolve()

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

VIDEO_EXTENSIONS = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"]

# Exactly NUM_FRAMES frames are sampled uniformly over the full video.
NUM_FRAMES = 256

MAX_NEW_TOKENS = 2048

QUESTION_COLUMN = "Questions"

PROMPT_TEMPLATE = (
    "Answer the following question "
    "using the information available "
    "in the video.\n\n"
    "Question: {question}\n\n"
)


# ============================================================
# GPU / SPEED CONFIG
# ============================================================

GPU_MEMORY_UTILIZATION = 0.90

# Questions of this many videos are decoded together in one batch.
# 4 videos x 10 questions = 40 sequences -> far better GPU use than
# 10 at a time, while keeping host RAM for decoded frames bounded.
VIDEOS_PER_BATCH = 4

# Room for the 256-frame video tokens + question + 2048 output tokens.
MAX_MODEL_LEN = 65536

# Bigger prefill chunks -> faster processing of the long video prompt.
MAX_NUM_BATCHED_TOKENS = 16384

# Videos decoded in parallel on CPU while the GPUs are busy.
DECODE_WORKERS = 4

# "opencv" (default, no extra system deps) or "torchcodec" (faster
# seeking, needs FFmpeg shared libs on the machine).
DECODE_BACKEND = os.environ.get("DECODE_BACKEND", "opencv")


# ============================================================
# HELPERS
# ============================================================

def count_gpus():

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip()])

    # nvidia-smi instead of torch.cuda so CUDA is not initialized in
    # the parent process before vLLM starts its workers.
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0

    return len([line for line in out.splitlines() if line.startswith("GPU ")])


def pick_tensor_parallel(num_gpus):

    # Qwen3-VL-32B has 64 attention / 8 KV heads -> TP must be 1, 2, 4 or 8.
    tp = 1
    while tp * 2 <= min(num_gpus, 8):
        tp *= 2
    return tp


def find_videos():

    videos = []
    for extension in VIDEO_EXTENSIONS:
        videos.extend(VIDEO_DIR.glob(extension))
    return sorted(videos, key=lambda x: x.name)


def load_output_df(questions_df, output_file):

    fresh = pd.DataFrame({
        "question": questions_df[QUESTION_COLUMN].astype(str),
        "answer": "",
    })

    if not output_file.exists():
        return fresh

    output_df = pd.read_csv(output_file, keep_default_na=False)

    if (
        "question" not in output_df.columns
        or "answer" not in output_df.columns
        or len(output_df) != len(questions_df)
    ):
        print(f"[WARNING] {output_file.name} does not match question CSV. Reinitializing.")
        return fresh

    output_df = output_df[["question", "answer"]].copy()
    output_df["answer"] = output_df["answer"].astype(object)

    return output_df


def build_jobs(videos):
    """Validate every video/question pair BEFORE the model is loaded."""

    jobs = []

    for video_path in videos:

        video_name = video_path.stem
        question_file = QUESTION_DIR / f"{video_name}.csv"
        output_file = OUTPUT_DIR / f"{video_name}_answers.csv"

        if not question_file.exists():
            print(f"[WARNING] No question file for {video_path.name} (expected {question_file.name})")
            continue

        questions_df = pd.read_csv(question_file)

        if QUESTION_COLUMN not in questions_df.columns:
            print(
                f"[ERROR] Column '{QUESTION_COLUMN}' not found in {question_file.name}. "
                f"Available columns: {list(questions_df.columns)}"
            )
            continue

        output_df = load_output_df(questions_df, output_file)

        # RESUME SUPPORT: only questions without an answer
        pending = [
            idx for idx in range(len(questions_df))
            if str(output_df.loc[idx, "answer"]).strip() == ""
        ]

        if not pending:
            print(f"[SKIP] {video_name}: all {len(questions_df)} questions already answered")
            continue

        jobs.append({
            "video_path": video_path,
            "name": video_name,
            "questions_df": questions_df,
            "output_df": output_df,
            "output_file": output_file,
            "pending": pending,
        })

    return jobs


def decode_video(video_path):
    """Decode NUM_FRAMES frames once per video (not once per question).

    Uses vLLM's own uniform-sampling loader, so frames and timestamp
    metadata match what vLLM produces for a file:// video. (vLLM's
    default loader for Qwen3-VL ignores num_frames and samples at 2 fps,
    up to 768 frames - this pins it to exactly NUM_FRAMES.)
    """

    t0 = time.time()

    # Older vLLM versions have no `backend` argument (always OpenCV).
    extra = {} if DECODE_BACKEND == "opencv" else {"backend": DECODE_BACKEND}

    frames, metadata = UniformVideoLoader.load_bytes(
        video_path.read_bytes(),
        num_frames=NUM_FRAMES,
        **extra,
    )

    # Frames are already sampled here; the HF processor must not
    # re-sample them (it would otherwise do so for videos that have
    # <= NUM_FRAMES frames in total).
    metadata["do_sample_frames"] = False

    return frames, metadata, time.time() - t0


def build_prompt(processor, question):

    # Video comes FIRST so every question of the same video shares an
    # identical prefix -> vLLM prefix cache reuses the video KV cache.
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


def make_request(processor, job, question):

    return {
        "prompt": build_prompt(processor, question),
        "multi_modal_data": {"video": (job["frames"], job["metadata"])},
        # Same id for every question of a video: vLLM skips re-hashing
        # the frames and re-running the HF video processor.
        "multi_modal_uuids": {"video": [job["uuid"]]},
    }


def save_job(job, outputs):

    output_df = job["output_df"]
    questions_df = job["questions_df"]

    for idx, out in zip(job["pending"], outputs):

        answer = out.outputs[0].text.strip() if out.outputs else ""
        output_df.loc[idx, "answer"] = answer

        print(f"\n[{job['name']}] Q{idx + 1}: {questions_df.loc[idx, QUESTION_COLUMN]}")
        print(f"A: {answer}")

        if out.outputs and out.outputs[0].finish_reason == "length":
            print(f"[WARNING] answer hit MAX_NEW_TOKENS={MAX_NEW_TOKENS} and was cut off")

    output_df.to_csv(job["output_file"], index=False)

    print(
        f"\nSaved: {job['output_file'].name} | "
        f"prompt tokens: {len(outputs[0].prompt_token_ids)}"
    )


def run_batch(llm, processor, sampling_params, batch):

    t0 = time.time()

    # --------------------------------------------------------
    # WARM-UP: prefill each video once (1 output token) so the
    # full batch hits the prefix cache for every question instead
    # of prefilling the same video N times in parallel.
    # --------------------------------------------------------

    warmup = [
        make_request(
            processor, job,
            str(job["questions_df"].loc[job["pending"][0], QUESTION_COLUMN]),
        )
        for job in batch
    ]

    llm.generate(
        warmup,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        use_tqdm=False,
    )

    t1 = time.time()
    print(f"Video prefill ({len(batch)} videos): {t1 - t0:.2f}s")

    # --------------------------------------------------------
    # ALL QUESTIONS OF ALL VIDEOS IN THIS BATCH AT ONCE
    # --------------------------------------------------------

    requests = [
        make_request(processor, job, str(job["questions_df"].loc[idx, QUESTION_COLUMN]))
        for job in batch
        for idx in job["pending"]
    ]

    outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=True)

    t2 = time.time()

    start = 0
    for job in batch:
        n = len(job["pending"])
        save_job(job, outputs[start:start + n])
        start += n

    generated = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)

    print(
        f"\nBatch done: {len(requests)} questions in {t2 - t0:.2f}s | "
        f"{generated / max(t2 - t1, 1e-6):.1f} output tok/s"
    )


def process_batch(llm, processor, sampling_params, batch):

    try:
        run_batch(llm, processor, sampling_params, batch)
        return
    except Exception as e:
        if len(batch) == 1:
            print(f"\n[ERROR] Video: {batch[0]['name']}\nError: {repr(e)}")
            return
        print(f"\n[ERROR] Batch failed ({repr(e)}). Retrying videos one by one.")

    # One bad video must not cost the answers of the others.
    for job in batch:
        process_batch(llm, processor, sampling_params, [job])


def write_combined_csv():

    frames = []
    for output_file in sorted(OUTPUT_DIR.glob("*_answers.csv")):
        df = pd.read_csv(output_file, keep_default_na=False)
        df.insert(0, "video", output_file.stem.removesuffix("_answers"))
        df.insert(1, "question_index", range(1, len(df) + 1))
        frames.append(df)

    if frames:
        combined = OUTPUT_DIR / "all_answers.csv"
        pd.concat(frames, ignore_index=True).to_csv(combined, index=False)
        print(f"Combined output: {combined}")


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME,
                        help="e.g. Qwen/Qwen3-VL-32B-Instruct-FP8 for ~2x less memory")
    parser.add_argument("--tp", type=int, default=None,
                        help="tensor parallel size (default: all GPUs, power of 2)")
    parser.add_argument("--videos-per-batch", type=int, default=VIDEOS_PER_BATCH)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N videos (smoke test)")
    return parser.parse_args()


def main():

    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # PRE-FLIGHT: fail fast before spending minutes loading the model
    # --------------------------------------------------------

    videos = find_videos()
    if args.limit:
        videos = videos[:args.limit]

    print(f"Found {len(videos)} videos in {VIDEO_DIR}")
    jobs = build_jobs(videos)
    total_pending = sum(len(j["pending"]) for j in jobs)
    print(f"Videos to process: {len(jobs)} | questions pending: {total_pending}")

    if not jobs:
        print("Nothing to do.")
        write_combined_csv()
        return

    num_gpus = count_gpus()
    if num_gpus == 0:
        raise RuntimeError("No NVIDIA GPU visible (nvidia-smi -L returned nothing).")

    tp = args.tp or pick_tensor_parallel(num_gpus)

    print("=" * 80)
    print("Qwen3-VL-32B-Instruct + vLLM")
    print("=" * 80)
    print(f"Model              : {args.model}")
    print(f"GPUs visible       : {num_gpus}")
    print(f"Tensor parallel    : {tp}")
    print(f"Video frames       : {NUM_FRAMES} ({DECODE_BACKEND} decode)")
    print(f"Max new tokens     : {MAX_NEW_TOKENS}")
    print(f"Videos per batch   : {args.videos_per_batch}")
    print(f"Max model length   : {MAX_MODEL_LEN}")
    print("=" * 80)

    if tp == 1 and "FP8" not in args.model.upper():
        print(
            "[WARNING] 32B in bf16 on a single GPU leaves almost no KV cache "
            "unless it has >=141GB (H200/B200). Use more GPUs or "
            "--model Qwen/Qwen3-VL-32B-Instruct-FP8."
        )

    # Decode upcoming videos on CPU while the model loads / GPUs run.
    decoder = ThreadPoolExecutor(max_workers=DECODE_WORKERS)
    futures = {}

    def batch_jobs(batch_idx):
        return jobs[batch_idx * args.videos_per_batch:(batch_idx + 1) * args.videos_per_batch]

    def prefetch(batch_idx):
        for job in batch_jobs(batch_idx):
            if job["name"] not in futures:
                futures[job["name"]] = decoder.submit(decode_video, job["video_path"])

    prefetch(0)

    t_model_start = time.time()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=max(64, args.videos_per_batch * 20),
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,

        # Reuse the video KV cache across questions of the same video.
        enable_prefix_caching=True,

        # Run the vision encoder data-parallel on every GPU
        # (faster than splitting the small ViT with tensor parallel).
        mm_encoder_tp_mode="data",

        limit_mm_per_prompt={"video": 1, "image": 0},

        # Keeps processed videos cached across the questions of a batch.
        mm_processor_cache_gb=8,

        # CUDA graphs ON (no enforce_eager) -> much faster decoding.
    )

    processor = AutoProcessor.from_pretrained(args.model)

    print(f"\nModel loaded in {time.time() - t_model_start:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    t_start = time.time()
    num_batches = (len(jobs) + args.videos_per_batch - 1) // args.videos_per_batch

    for batch_idx in range(num_batches):

        prefetch(batch_idx + 1)

        print("\n" + "=" * 80)
        print(f"BATCH {batch_idx + 1}/{num_batches}: {', '.join(j['name'] for j in batch_jobs(batch_idx))}")
        print("=" * 80)

        batch = []
        for job in batch_jobs(batch_idx):
            try:
                frames, metadata, decode_s = futures.pop(job["name"]).result()
            except Exception as e:
                print(f"[ERROR] Could not decode {job['video_path'].name}: {repr(e)}")
                continue

            stat = job["video_path"].stat()
            job["frames"] = frames
            job["metadata"] = metadata
            job["uuid"] = f"{job['name']}:{stat.st_size}:{stat.st_mtime_ns}:{NUM_FRAMES}"
            batch.append(job)

            print(
                f"{job['name']}: {len(metadata['frames_indices'])} frames "
                f"{frames.shape[2]}x{frames.shape[1]}, "
                f"{metadata['duration']:.1f}s video, decoded in {decode_s:.1f}s, "
                f"{len(job['pending'])} questions"
            )

        if batch:
            process_batch(llm, processor, sampling_params, batch)

        # Free decoded frames before the next batch.
        for job in batch:
            job.pop("frames", None)

        elapsed = time.time() - t_start
        print(
            f"\nProgress: {batch_idx + 1}/{num_batches} batches | "
            f"{elapsed / 60:.1f} min elapsed | "
            f"~{elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1) / 60:.1f} min left"
        )

    decoder.shutdown(wait=False, cancel_futures=True)

    print("\n" + "=" * 80)
    print(f"ALL VIDEOS COMPLETED in {(time.time() - t_start) / 60:.1f} min")
    print("=" * 80)

    write_combined_csv()


# Required: vLLM spawns one worker process per GPU.
if __name__ == "__main__":
    main()
