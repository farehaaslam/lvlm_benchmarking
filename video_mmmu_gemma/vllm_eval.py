"""
Run the official google/gemma-4-12B-it (BF16, not quantized) on Video-MMMU (all 900 questions) with vLLM,
sending the VIDEO itself as input (vLLM's Gemma 4 video path), the official lmms-eval prompts and scoring
(videommmu_official.py), and the same generation settings and budget as gurrt/automation/llama_inference.py.

Why this is fast and native_sampling.py is not: native_sampling.py runs HF `generate` one question at a time.
Here vLLM batches many questions at once (continuous batching, paged KV cache, CUDA graphs) and the 3
questions of a video share the video prefix through prefix caching.

Copied from llama_inference.py:
  temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, seed 0,
  thinking on, thinking budget 16378 tokens; when it runs out the same message
  "I have thought enough. Now I will give the final answer." is forced and the thought is closed,
  max_tokens = 16378 + 4096, context = max_tokens + 24576.

Video: passed as a file:// video_url; vLLM decodes it (NUM_FRAMES uniform frames) and Gemma 4's own video
processor turns it into tokens. Adaptation: the dataset's question image is attached right after the
question text (video -> question -> image).

GPU memory: BF16 12B weights alone are ~24 GB.
  RTX 5090 (32 GB): fits on one GPU, ~6 GB left for KV cache.
  RTX 4090 (24 GB): does NOT fit on one card -> use 2 cards with --tp 2.
Several GPUs: one process per GPU (or per pair with --tp 2), each on its own shard, e.g. 4 x 5090:
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i nohup uv run --no-sync python vllm_eval.py \\
        --shard_id $i --num_shards 4 > logs/vllm_$i.log 2>&1 & done
    then: uv run python vllm_eval.py --summary_only      (or: python score.py --out_dir <OUT_DIR>)

Output: one CSV per video in OUT_DIR, same columns as native_sampling.py (+ output_tokens, hit_budget,
reasoning), so score.py works. Finished videos are skipped on rerun. Test first with --limit_videos 3.
"""
import os

# vLLM workers must be spawned, not forked (set before vLLM is imported)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# FlashInfer JIT-compiles its top-k/top-p sampler with the system nvcc (/usr/local/cuda = 12.8 here), which
# can't target RTX 5090 (SM 12.0 needs >= 12.9) -> "FlashInfer requires GPUs with sm75". Use vLLM's torch sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# ============================ CONFIG: EDIT THESE ============================
REPO_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = f"{REPO_DIR}/dataset/videommmu"
VIDEO_DIR      = DATA_DIR
OUT_DIR        = f"{REPO_DIR}/results/videommmu_gemma4_12b_vllm"
MODEL_ID       = "google/gemma-4-12B-it"
NUM_FRAMES     = 32                           # frames vLLM decodes from each video
VIDEOS_PER_BATCH = 16                         # 16 videos x 3 questions per generate call (bounds host RAM)
GPU_MEMORY_UTILIZATION = 0.92
LIMIT_VIDEOS   = 0
# ============================================================================

# ---- identical to gurrt/automation/llama_inference.py ----
REASONING_BUDGET = 16378
REASONING_BUDGET_MESSAGE = "I have thought enough. Now I will give the final answer."
MAX_TOKENS = REASONING_BUDGET + 4096
MAX_MODEL_LEN = MAX_TOKENS + 24576
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, seed=0)
# -----------------------------------------------------------

# Gemma 4 thinking delimiters (vllm/reasoning/gemma4_utils.py)
THINK_START, THINK_END = "<|channel>", "<channel|>"

import argparse, base64, io, json, time, traceback

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

from vllm.multimodal.media.connector import MediaConnector

import videommmu_official as vm
from native_sampling import TRACKS, build_video_index, load_image, load_questions, natural_key, summarize


# vLLM decodes the video separately for each of a video's 3 questions (~5-10 s of CPU each, the GPU idles
# meanwhile). Decode once per URL and reuse; run_batch clears the cache so host RAM stays bounded to one batch.
_video_cache = {}
_fetch_video = MediaConnector.fetch_video


def _cached_fetch_video(self, video_url, **kwargs):
    key = (video_url, repr(sorted(kwargs.items())))     # repr: vLLM passes video_processor as a dict
    if key not in _video_cache:
        _video_cache[key] = _fetch_video(self, video_url, **kwargs)
    return _video_cache[key]


MediaConnector.fetch_video = _cached_fetch_video


def image_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_conversation(vpath, doc, track):
    # video, then the question with its Adaptation image attached right after it
    content = [
        {"type": "video_url", "video_url": {"url": "file://" + os.path.abspath(vpath)}},
        {"type": "text", "text": vm.doc_to_text(doc, track)},
    ]
    image = load_image(doc.get("image")) if track == "Adaptation" else None
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
    return [{"role": "user", "content": content}], image is not None


def split_thinking(text):
    """(reasoning, answer). A thought that never closed (hit max_tokens) leaves no answer."""
    if THINK_START in text and THINK_END not in text:
        return text, ""
    if THINK_END in text:
        reasoning, answer = text.rsplit(THINK_END, 1)
        reasoning = reasoning.split(THINK_START, 1)[-1]
        reasoning = reasoning[len("thought"):] if reasoning.startswith("thought") else reasoning
        return reasoning.strip(), vm.strip_special_tokens(answer)
    answer = text[len("thought\n"):] if text.startswith("thought\n") else text
    return "", vm.strip_special_tokens(answer)


def run_batch(llm, params, batch, by_id, vid_index, out_dir):
    convs, meta = [], []
    for vid in batch:
        for track in TRACKS:
            conv, used_image = build_conversation(vid_index[vid], by_id[vid][track], track)
            convs.append(conv)
            meta.append((vid, track, used_image))

    try:
        outs = llm.chat(convs, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    finally:
        _video_cache.clear()

    rows = {}
    for (vid, track, used_image), out in zip(meta, outs):
        doc = by_id[vid][track]
        o = out.outputs[0]
        reasoning, resp = split_thinking(o.text)
        parsed, correct = vm.score(doc, resp)
        subject = vm.extract_subset_name(vid)
        rows.setdefault(vid, []).append({
            "id": vid, "track": track, "subject": subject, "domain": vm.SUB_CAT2DOMAIN.get(subject, ""),
            "question_type": doc["question_type"], "qa_type": doc.get("qa_type", ""),
            "answer": doc["answer"], "parsed_pred": parsed if isinstance(parsed, str) else json.dumps(parsed),
            "correct": int(correct), "input_tokens": len(out.prompt_token_ids),
            "output_tokens": len(o.token_ids), "hit_budget": int(REASONING_BUDGET_MESSAGE in reasoning),
            "used_question_image": int(used_image), "response": resp, "reasoning": reasoning,
        })
    for vid, r in rows.items():
        pd.DataFrame(r).to_csv(os.path.join(out_dir, f"{vid}.csv"), index=False)
        print(f"{vid}: " + ", ".join(f"{x['track'][:4]}={x['correct']}" for x in r) +
              f"  in_tok={r[0]['input_tokens']}  out_tok={[x['output_tokens'] for x in r]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--video_dir", default=VIDEO_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (2 for BF16 on 24 GB cards)")
    ap.add_argument("--frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--videos_per_batch", type=int, default=VIDEOS_PER_BATCH)
    ap.add_argument("--max_model_len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--limit_videos", type=int, default=LIMIT_VIDEOS)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    by_id = load_questions(a.data_dir)
    ids = sorted(by_id, key=natural_key)
    vid_index = build_video_index(a.video_dir)
    print(f"{len(ids)} videos in the question files, {len(vid_index)} video files indexed, "
          f"{sum(v not in vid_index for v in ids)} missing")
    if a.summary_only:
        summarize(a.out_dir, by_id)
        return
    if a.limit_videos:
        ids = ids[: a.limit_videos]
    ids = ids[a.shard_id :: a.num_shards]
    missing = [v for v in ids if v not in vid_index]
    todo = [v for v in ids if v in vid_index and not os.path.exists(os.path.join(a.out_dir, f"{v}.csv"))]
    print(f"shard {a.shard_id}/{a.num_shards}: {len(ids)} videos, {len(todo)} to run, {len(missing)} without a video file")
    if not todo:
        summarize(a.out_dir, by_id)
        return

    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        tensor_parallel_size=a.tp,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=True,                     # the 3 questions of a video share the video prefix
        limit_mm_per_prompt={"video": 1, "image": 1},
        # explicit backend (= vLLM's default) skips vLLM 0.30's per-processor backend lookup, which crashes
        # with "unhashable type: 'dict'" because Gemma 4's processor_config nests video_processor as a dict
        media_io_kwargs={"video": {"num_frames": a.frames, "video_backend": "opencv"}},
        allowed_local_media_path=os.path.abspath(a.video_dir),
        # thinking budget: after REASONING_BUDGET thought tokens, force the budget message and close the thought
        reasoning_config=ReasoningConfig(
            reasoning_start_str=THINK_START,
            reasoning_end_str=f"\n\n{REASONING_BUDGET_MESSAGE}{THINK_END}",
        ),
    )
    params = SamplingParams(
        **SAMPLING,
        max_tokens=MAX_TOKENS,
        thinking_token_budget=REASONING_BUDGET,
        skip_special_tokens=False,                      # keep the thought delimiters so they can be split off
    )

    t0, failed = time.time(), []
    batches = [todo[i : i + a.videos_per_batch] for i in range(0, len(todo), a.videos_per_batch)]
    for b, batch in enumerate(batches, 1):
        try:
            run_batch(llm, params, batch, by_id, vid_index, a.out_dir)
        except Exception as e:
            # one bad video must not cost the rest of the batch
            print(f"batch {b} failed ({type(e).__name__}: {e}), retrying its videos one by one", flush=True)
            for vid in batch:
                try:
                    run_batch(llm, params, [vid], by_id, vid_index, a.out_dir)
                except Exception as e2:
                    failed.append(vid)
                    traceback.print_exc()
                    print(f"{vid}: NOT SAVED ({type(e2).__name__}: {e2}) - retried on next run", flush=True)
        el = (time.time() - t0) / 60
        print(f"--- batch {b}/{len(batches)} done, {el:.1f} min elapsed, ~{el / b * (len(batches) - b):.0f} min left",
              flush=True)

    if failed:
        print(f"\n{len(failed)} videos failed and were not saved: {failed}")
    if a.num_shards == 1:
        summarize(a.out_dir, by_id)


if __name__ == "__main__":
    main()
