import gc
import time
from pathlib import Path

import pandas as pd
import torch

from tqdm import tqdm

from transformers import AutoProcessor

from vllm import LLM, SamplingParams


#to run this script 
# CUDA_VISIBLE_DEVICES=0,1,2 python qwen32B_vllm.py   when tensor parallel size is 3


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"


# ============================================================
# VIDEO CONFIG
# ============================================================

# IMPORTANT:
#
# We DO NOT sample frames ourselves.
#
# The COMPLETE video is passed to vLLM.
# vLLM internally samples exactly 256 frames.
#
NUM_FRAMES = 256


# ============================================================
# GENERATION CONFIG
# ============================================================

MAX_NEW_TOKENS = 512

QUESTION_COLUMN = "Questions"


# ============================================================
# GPU CONFIG
# ============================================================

# Example:
#
# 2 GPUs:
#     CUDA_VISIBLE_DEVICES=0,1
#     TENSOR_PARALLEL_SIZE = 2
#
# 3 GPUs:
#     CUDA_VISIBLE_DEVICES=0,1,2
#     TENSOR_PARALLEL_SIZE = 3

TENSOR_PARALLEL_SIZE = 3

# Fraction of each GPU's memory vLLM can use.
GPU_MEMORY_UTILIZATION = 0.90

# Number of concurrent sequences.
#
# Start with 1 for 256-frame Qwen3-VL-32B.
# Increase to 2/4 only after checking VRAM.
MAX_NUM_SEQS = 1

# Qwen3-VL supports very long contexts.
# For this benchmark, 32768 is a reasonable starting point.
MAX_MODEL_LEN = 32768


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

print("=" * 80)
print("Qwen3-VL-32B-Instruct + vLLM")
print("=" * 80)

print(f"Model              : {MODEL_NAME}")
print(f"Video directory    : {VIDEO_DIR}")
print(f"Question directory : {QUESTION_DIR}")
print(f"Output directory   : {OUTPUT_DIR}")

print(f"\nVideo frames       : {NUM_FRAMES}")
print(f"Tensor parallel    : {TENSOR_PARALLEL_SIZE}")
print(f"GPU memory         : {GPU_MEMORY_UTILIZATION}")
print(f"Max sequences      : {MAX_NUM_SEQS}")
print(f"Max model length   : {MAX_MODEL_LEN}")

print("=" * 80)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading Qwen3-VL processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded.")


# ============================================================
# LOAD VLLM MODEL
# ============================================================

print("\nLoading Qwen3-VL-32B with vLLM...")

t_model_start = time.time()

llm = LLM(

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model=MODEL_NAME,

    # --------------------------------------------------------
    # Multi-GPU tensor parallelism
    # --------------------------------------------------------

    tensor_parallel_size=TENSOR_PARALLEL_SIZE,

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    max_model_len=MAX_MODEL_LEN,

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,

    # --------------------------------------------------------
    # Concurrent sequences
    # --------------------------------------------------------

    max_num_seqs=MAX_NUM_SEQS,

    # --------------------------------------------------------
    # Allow one video per request
    # --------------------------------------------------------

    limit_mm_per_prompt={
        "video": 1
    },

    # --------------------------------------------------------
    # VIDEO SAMPLING
    #
    # The COMPLETE video is passed to vLLM.
    #
    # vLLM internally samples exactly NUM_FRAMES frames.
    # --------------------------------------------------------

    media_io_kwargs={
        "video": {
            "num_frames": NUM_FRAMES
        }
    },

    # --------------------------------------------------------
    # Start conservatively.
    #
    # Remove this if your vLLM/model combination works
    # correctly with CUDA graphs.
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

    # Deterministic generation
    temperature=0.0,

    # Maximum generated tokens
    max_tokens=MAX_NEW_TOKENS,
)


# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted(
    VIDEO_DIR.glob("*.mp4"),
    key=lambda x: x.name
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

    print("\n" + "=" * 80)
    print(f"VIDEO: {video_name}")
    print("=" * 80)

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

            # ------------------------------------------------
            # Safety check:
            # make sure output has enough rows
            # ------------------------------------------------

            if len(output_df) != len(questions_df):

                print(
                    "[WARNING] Existing output row count "
                    "does not match question CSV."
                )

                output_df = pd.DataFrame({

                    "question": questions_df[
                        QUESTION_COLUMN
                    ].astype(str),

                    "answer": ""
                })

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
        # RESUME SUPPORT
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
        # GET QUESTION
        # ----------------------------------------------------

        question = str(
            questions_df.loc[
                idx,
                QUESTION_COLUMN
            ]
        )

        print(
            f"\nQuestion {idx + 1}/"
            f"{len(questions_df)}"
        )

        print(
            f"Question: {question}"
        )

        try:

            # =================================================
            # QWEN MESSAGE
            # =================================================

            messages = [
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "video",

                            # COMPLETE VIDEO PATH
                            #
                            # We are NOT extracting frames.
                            #
                            # vLLM receives this complete video
                            # and samples NUM_FRAMES internally.

                            "video": str(video_path),
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


            # =================================================
            # APPLY QWEN CHAT TEMPLATE
            # =================================================

            t0 = time.time()

            prompt = processor.apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=True,
            )

            t1 = time.time()


            # =================================================
            # VLLM REQUEST
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

            outputs = llm.generate(

                [request],

                sampling_params=sampling_params,
            )

            t2 = time.time()


            # =================================================
            # EXTRACT ANSWER
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

            prompt_time = t1 - t0

            generation_time = t2 - t1

            total_time = t2 - t0

            print(
                f"Prompt preparation : "
                f"{prompt_time:.2f}s"
            )

            print(
                f"Generation         : "
                f"{generation_time:.2f}s"
            )

            print(
                f"Total              : "
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
            # Save completed answers
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

print("\n" + "=" * 80)
print("ALL VIDEOS COMPLETED")
print("=" * 80)