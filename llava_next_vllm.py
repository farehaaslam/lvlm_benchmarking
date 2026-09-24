import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
import re

from vllm import LLM, SamplingParams


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "llava-hf/LLaVA-NeXT-Video-7B-hf"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

# ============================================================
# VIDEO CONFIG
# ============================================================

# vLLM will receive the COMPLETE video and internally sample
# exactly this many frames.
NUM_FRAMES = 256

# ============================================================
# GENERATION CONFIG
# ============================================================

MAX_NEW_TOKENS = 2048

TEMPERATURE = 0.0

QUESTION_COLUMN = "Questions"


# ============================================================
# GPU CONFIG
# ============================================================

# Number of GPUs to use.
#
# Example:
#   1 GPU -> 1
#   2 GPUs -> 2
#   3 GPUs -> 3
#
# This uses tensor parallelism.
TENSOR_PARALLEL_SIZE = 3

# Keep this conservative initially.
GPU_MEMORY_UTILIZATION = 0.90

# Maximum number of simultaneous requests.
#
# IMPORTANT:
# LLaVA-NeXT-Video with 256 frames is memory-heavy.
# Start with 1 and increase only after confirming VRAM.
MAX_NUM_SEQS = 1

# Context length.
MAX_MODEL_LEN = 8192


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PRINT CONFIG
# ============================================================

print("=" * 70)
print("LLaVA-NeXT-Video + vLLM")
print("=" * 70)

print(f"Model              : {MODEL_NAME}")
print(f"Video directory    : {VIDEO_DIR}")
print(f"Question directory : {QUESTION_DIR}")
print(f"Output directory   : {OUTPUT_DIR}")
print(f"Frames per video   : {NUM_FRAMES}")
print(f"Tensor parallel    : {TENSOR_PARALLEL_SIZE}")
print(f"GPU memory util.   : {GPU_MEMORY_UTILIZATION}")
print(f"Max sequences      : {MAX_NUM_SEQS}")
print(f"Max model length   : {MAX_MODEL_LEN}")
print("=" * 70)


# ============================================================
# LOAD VLLM MODEL
# ============================================================

print("\nLoading LLaVA-NeXT-Video with vLLM...")

t_model_start = time.time()

llm = LLM(
    model=MODEL_NAME,

    # --------------------------------------------------------
    # Multi-GPU tensor parallelism
    # --------------------------------------------------------
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,

    # --------------------------------------------------------
    # Model context
    # --------------------------------------------------------
    max_model_len=MAX_MODEL_LEN,

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,

    # --------------------------------------------------------
    # Number of concurrent sequences
    # --------------------------------------------------------
    max_num_seqs=MAX_NUM_SEQS,

    # --------------------------------------------------------
    # Tell vLLM that each request contains one video
    # --------------------------------------------------------
    limit_mm_per_prompt={
        "video": 1
    },

    # --------------------------------------------------------
    # Tell vLLM how many frames to extract from the COMPLETE
    # video.
    #
    # The video itself is passed to vLLM.
    # vLLM performs the frame sampling.
    # --------------------------------------------------------
    media_io_kwargs={
        "video": {
            "num_frames": NUM_FRAMES
        }
    },

    # --------------------------------------------------------
    # Optional: allow vLLM to use eager execution if needed.
    # Remove this if your installation works without it.
    # --------------------------------------------------------
    enforce_eager=True,
)

t_model_end = time.time()

print(
    f"\nModel loaded in "
    f"{t_model_end - t_model_start:.2f}s"
)


# ============================================================
# SAMPLING PARAMETERS
# ============================================================

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_NEW_TOKENS,
)


# ============================================================
# FIND VIDEOS
# ============================================================
videos = sorted(
    VIDEO_DIR.glob("*.mp4"),
    key=lambda x: (
        int(re.match(r"(\d+)", x.stem).group(1)),
        0 if re.match(r"^\d+$", x.stem) else 1
    )
)

print(
    f"\nFound {len(videos)} videos."
)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    video_name = video_path.stem

    question_file = (
        QUESTION_DIR /
        f"{video_name}.csv"
    )

    output_file = (
        OUTPUT_DIR /
        f"{video_name}_answers.csv"
    )

    print("\n" + "=" * 70)
    print(f"VIDEO: {video_name}")
    print("=" * 70)

    # ========================================================
    # CHECK QUESTION CSV
    # ========================================================

    if not question_file.exists():

        print(
            f"[WARNING] Question file not found: "
            f"{question_file}"
        )

        return

    questions_df = pd.read_csv(
        question_file
    )

    if QUESTION_COLUMN not in questions_df.columns:

        print(
            f"[ERROR] Column '{QUESTION_COLUMN}' "
            f"not found in {question_file}"
        )

        print(
            f"Available columns: "
            f"{list(questions_df.columns)}"
        )

        return

    # ========================================================
    # CREATE / LOAD OUTPUT
    # ========================================================

    if output_file.exists():

        output_df = pd.read_csv(
            output_file
        )

        print(
            f"Existing output found: "
            f"{len(output_df)} rows"
        )

        # ----------------------------------------------------
        # Make sure output structure is valid
        # ----------------------------------------------------

        if (
            "question" not in output_df.columns
            or "answer" not in output_df.columns
        ):

            print(
                "[WARNING] Existing output has "
                "incorrect columns."
            )

            output_df = pd.DataFrame({
                "question": questions_df[
                    QUESTION_COLUMN
                ].astype(str),

                "answer": ""
            })

        else:

            output_df = output_df[
                ["question", "answer"]
            ]

    else:

        output_df = pd.DataFrame({

            "question": questions_df[
                QUESTION_COLUMN
            ].astype(str),

            "answer": ""
        })

    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for idx in tqdm(
        range(len(questions_df)),
        desc=f"Questions - {video_name}"
    ):

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        existing_answer = (
            output_df.loc[idx, "answer"]
        )

        if (
            pd.notna(existing_answer)
            and str(existing_answer).strip() != ""
        ):

            continue

        # ----------------------------------------------------
        # Get question
        # ----------------------------------------------------

        question = str(
            questions_df.loc[
                idx,
                QUESTION_COLUMN
            ]
        )

        print(
            f"\n[{idx + 1}/{len(questions_df)}]"
        )

        print(
            f"Question: {question}"
        )

        try:

            # =================================================
            # PROMPT
            # =================================================

            # LLaVA-NeXT-Video uses:
            #
            # USER: <video>
            # question
            # ASSISTANT:
            #
            prompt = (
                "USER: <video>\n"
                "Answer the following question using "
                "the information available in the video.\n\n"
                f"Question: {question}\n\n"
                "ASSISTANT:"
            )

            # =================================================
            # vLLM REQUEST
            # =================================================

            request = {
                "prompt": prompt,

                "multi_modal_data": {
                    "video": str(video_path)
                }
            }

            # =================================================
            # GENERATION
            # =================================================

            t0 = time.time()

            outputs = llm.generate(
                [request],
                sampling_params=sampling_params,
            )

            t1 = time.time()

            # =================================================
            # EXTRACT RESPONSE
            # =================================================

            if (
                outputs
                and outputs[0].outputs
            ):

                answer = (
                    outputs[0]
                    .outputs[0]
                    .text
                    .strip()
                )

            else:

                answer = ""

            # =================================================
            # TIMING
            # =================================================

            total_time = t1 - t0

            print(
                f"Generation: "
                f"{total_time:.2f}s"
            )

            print(
                f"Answer: {answer}"
            )

            # =================================================
            # SAVE ANSWER
            # =================================================

            output_df.loc[
                idx,
                "answer"
            ] = answer

            # Save after EVERY question
            output_df.to_csv(
                output_file,
                index=False
            )

        except Exception as e:

            print(
                "\n[ERROR]"
            )

            print(
                f"Video: {video_name}"
            )

            print(
                f"Question index: {idx}"
            )

            print(
                f"Question: {question}"
            )

            print(
                f"Error: {repr(e)}"
            )

            # ------------------------------------------------
            # Save whatever has completed
            # ------------------------------------------------

            output_df.to_csv(
                output_file,
                index=False
            )

            continue

        finally:

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # ========================================================
    # FINAL SAVE
    # ========================================================

    output_df[
        ["question", "answer"]
    ].to_csv(
        output_file,
        index=False
    )

    print(
        f"\nCompleted: {video_name}"
    )

    print(
        f"Output: {output_file}"
    )


# ============================================================
# MAIN LOOP
# ============================================================

for video_path in videos:

    process_video(
        video_path
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ALL VIDEOS COMPLETED")
print("=" * 70)