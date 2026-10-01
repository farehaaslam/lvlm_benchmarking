import os
import time
from pathlib import Path

import pandas as pd

from vllm import LLM, SamplingParams


# to run this script (all 4 GPUs, tensor parallel 4):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 uv run qwen32B_vllm.py


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DATASET_DIR = Path("dataset").resolve()

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

VIDEO_EXTENSIONS = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"]

# The COMPLETE video is passed to vLLM.
# vLLM internally samples exactly NUM_FRAMES frames.
NUM_FRAMES = 256

MAX_NEW_TOKENS = 2048

QUESTION_COLUMN = "Questions"


# ============================================================
# GPU / SPEED CONFIG
# ============================================================

# Tensor parallel: all 4 GPUs work on every token at the same time
# (HF device_map="auto" makes them take turns).
TENSOR_PARALLEL_SIZE = 4

GPU_MEMORY_UTILIZATION = 0.90

# All questions of one video are sent as a single batch.
# Videos have <= 20 questions, so 20 lets the whole batch decode together.
MAX_NUM_SEQS = 20

# Room for the 256-frame video tokens + question + 2048 output tokens.
# Check the "Prompt tokens" line printed per video; raise this if
# a video ever exceeds it.
MAX_MODEL_LEN = 65536

# Bigger prefill chunks -> faster processing of the long video prompt.
MAX_NUM_BATCHED_TOKENS = 16384


# ============================================================
# HELPERS
# ============================================================

def build_messages(video_path, question):

    # Video comes FIRST so every question of the same video shares
    # an identical prefix -> vLLM prefix cache reuses the video KV
    # cache instead of re-encoding / re-prefilling it per question.
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": video_path.as_uri()},
                },
                {
                    "type": "text",
                    "text": (
                        "Answer the following question "
                        "using the information available "
                        "in the video.\n\n"
                        f"Question: {question}\n\n"
                    ),
                },
            ],
        }
    ]


def load_output_df(questions_df, output_file):

    fresh = pd.DataFrame({
        "question": questions_df[QUESTION_COLUMN].astype(str),
        "answer": "",
    })

    if not output_file.exists():
        return fresh

    output_df = pd.read_csv(output_file)

    print(f"Existing output found: {len(output_df)} rows")

    if (
        "question" not in output_df.columns
        or "answer" not in output_df.columns
        or len(output_df) != len(questions_df)
    ):
        print("[WARNING] Existing output does not match question CSV. Reinitializing.")
        return fresh

    output_df = output_df[["question", "answer"]].copy()
    output_df["answer"] = output_df["answer"].astype(object)

    return output_df


def process_video(llm, sampling_params, video_path):

    video_name = video_path.stem

    question_file = QUESTION_DIR / f"{video_name}.csv"
    output_file = OUTPUT_DIR / f"{video_name}_answers.csv"

    print("\n" + "=" * 80)
    print(f"VIDEO: {video_name}")
    print("=" * 80)

    if not question_file.exists():
        print(f"[WARNING] Question file not found: {question_file}")
        return

    questions_df = pd.read_csv(question_file)

    if QUESTION_COLUMN not in questions_df.columns:
        print(f"[ERROR] Column '{QUESTION_COLUMN}' not found in {question_file}")
        print(f"Available columns: {list(questions_df.columns)}")
        return

    output_df = load_output_df(questions_df, output_file)

    # --------------------------------------------------------
    # RESUME SUPPORT: only questions without an answer
    # --------------------------------------------------------

    pending = [
        idx for idx in range(len(questions_df))
        if pd.isna(output_df.loc[idx, "answer"])
        or str(output_df.loc[idx, "answer"]).strip() == ""
    ]

    if not pending:
        print("All questions already answered. Skipping.")
        return

    print(f"Pending questions: {len(pending)}/{len(questions_df)}")

    conversations = [
        build_messages(
            video_path,
            str(questions_df.loc[idx, QUESTION_COLUMN]),
        )
        for idx in pending
    ]

    t0 = time.time()

    try:

        # ----------------------------------------------------
        # WARM-UP: prefill the video once (1 output token) so
        # the batch below hits the prefix cache for every
        # question instead of prefilling the video N times.
        # ----------------------------------------------------

        llm.chat(
            conversations[:1],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )

        t1 = time.time()

        print(f"Video decode + prefill: {t1 - t0:.2f}s")

        # ----------------------------------------------------
        # ALL QUESTIONS IN ONE BATCH
        # ----------------------------------------------------

        outputs = llm.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

    except Exception as e:

        print(f"\n[ERROR] Video: {video_name}\nError: {repr(e)}")
        return

    t2 = time.time()

    print(f"Prompt tokens: {len(outputs[0].prompt_token_ids)}")

    for idx, out in zip(pending, outputs):

        answer = out.outputs[0].text.strip() if out.outputs else ""

        output_df.loc[idx, "answer"] = answer

        print(f"\nQ{idx + 1}: {questions_df.loc[idx, QUESTION_COLUMN]}")
        print(f"A: {answer}")

    output_df.to_csv(output_file, index=False)

    generated = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)

    print(
        f"\nCompleted: {video_name} | "
        f"{len(pending)} questions in {t2 - t0:.2f}s | "
        f"{generated / (t2 - t1):.1f} output tok/s"
    )
    print(f"Output: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen3-VL-32B-Instruct + vLLM")
    print("=" * 80)
    print(f"Model              : {MODEL_NAME}")
    print(f"Video frames       : {NUM_FRAMES}")
    print(f"Max new tokens     : {MAX_NEW_TOKENS}")
    print(f"Tensor parallel    : {TENSOR_PARALLEL_SIZE}")
    print(f"Max sequences      : {MAX_NUM_SEQS}")
    print(f"Max model length   : {MAX_MODEL_LEN}")
    print("=" * 80)

    t_model_start = time.time()

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,

        # Reuse the video KV cache across questions of the same video.
        enable_prefix_caching=True,

        # Run the vision encoder data-parallel on every GPU
        # (faster than splitting the small ViT with tensor parallel).
        mm_encoder_tp_mode="data",

        limit_mm_per_prompt={"video": 1, "image": 0},

        # vLLM samples exactly NUM_FRAMES frames from the full video.
        media_io_kwargs={"video": {"num_frames": NUM_FRAMES}},

        # Needed to load videos from file:// URLs.
        allowed_local_media_path=str(VIDEO_DIR),

        # CUDA graphs ON (enforce_eager removed) -> much faster decoding.
    )

    print(f"\nModel loaded in {time.time() - t_model_start:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    videos = []
    for extension in VIDEO_EXTENSIONS:
        videos.extend(VIDEO_DIR.glob(extension))
    videos = sorted(videos, key=lambda x: x.name)

    print(f"\nFound {len(videos)} videos.")

    t_start = time.time()

    for video_path in videos:
        process_video(llm, sampling_params, video_path)

    print("\n" + "=" * 80)
    print(f"ALL VIDEOS COMPLETED in {(time.time() - t_start) / 60:.1f} min")
    print("=" * 80)


# Required: vLLM spawns one worker process per GPU.
if __name__ == "__main__":
    main()
