
import gc
import time
from pathlib import Path

import torch
import pandas as pd

from tqdm import tqdm
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

# Video sampling
NUM_FRAMES = 256

# Maximum generated tokens
MAX_NEW_TOKENS = 2048

# Question CSV column
QUESTION_COLUMN = "Questions"

# Supported video extensions
VIDEO_EXTENSIONS = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 70)
print("GPU INFORMATION")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):

    props = torch.cuda.get_device_properties(i)

    print(
        f"GPU {i}: {props.name} | "
        f"{props.total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen3-VL-32B-Instruct...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GPU MEMORY MONITOR
# ============================================================

def print_gpu_memory():

    for i in range(torch.cuda.device_count()):

        allocated = (
            torch.cuda.memory_allocated(i) / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(i) / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(i) / 1024**3
        )

        print(
            f"GPU {i}: "
            f"Allocated={allocated:.2f} GB | "
            f"Reserved={reserved:.2f} GB | "
            f"Peak={peak:.2f} GB"
        )


# ============================================================
# FIND VIDEOS
# ============================================================

videos = []

for extension in VIDEO_EXTENSIONS:
    videos.extend(VIDEO_DIR.glob(extension))

videos = sorted(
    videos,
    key=lambda x: x.name
)

print(f"\nFound {len(videos)} videos.")


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    video_name = video_path.stem

    question_file = (
        QUESTION_DIR / f"{video_name}.csv"
    )

    output_file = (
        OUTPUT_DIR / f"{video_name}_answers.csv"
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

        # Validate existing output
        if (
            "question" not in output_df.columns
            or "answer" not in output_df.columns
        ):

            print(
                "[WARNING] Existing output has "
                "incorrect columns. Reinitializing."
            )

            output_df = pd.DataFrame({
                "question": questions_df[
                    QUESTION_COLUMN
                ].astype(str),

                "answer": ""
            })

        else:

            # Retain only expected columns
            output_df = output_df[
                ["question", "answer"]
            ]

            # Ensure output has the same number of rows
            # as the question CSV.
            if len(output_df) != len(questions_df):

                print(
                    "[WARNING] Output row count differs "
                    "from question CSV."
                )

                print("Reinitializing output.")

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
        desc=f"Questions - {video_name}",
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

        print(f"\nQuestion {idx + 1}: {question}")

        inputs = None
        generated_ids = None
        generated_ids_trimmed = None

        try:

            # =================================================
            # MESSAGE
            # =================================================

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",

                            # Local video path
                            "video": str(video_path),

                            # Request 256 sampled frames
                            "nframes": NUM_FRAMES,

                            # Video resolution controls
                            # can be added here if needed
                        },
                        {
                            "type": "text",
                            "text": (
                                "Answer the following question "
                                "using the information available "
                                "in the video.\n\n"
                                f"Question: {question}\n\n"
                                "Provide a clear and accurate answer."
                            ),
                        },
                    ],
                }
            ]

            # =================================================
            # PROCESS INPUT
            # =================================================

            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            t1 = time.time()

            print(
                f"Processor: {t1 - t0:.2f}s"
            )

            # -------------------------------------------------
            # MOVE INPUTS
            # -------------------------------------------------

            inputs = inputs.to(
                model.device
            )

            # =================================================
            # GENERATE
            # =================================================

            generation_start = time.time()

            with torch.inference_mode():

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )

            generation_end = time.time()

            print(
                f"Generation: "
                f"{generation_end - generation_start:.2f}s"
            )

            print(
                f"Total: "
                f"{generation_end - t0:.2f}s"
            )

            # =================================================
            # REMOVE INPUT TOKENS
            # =================================================

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            # =================================================
            # DECODE
            # =================================================

            answer = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            print("\nANSWER:")
            print(answer)

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
                index=False,
            )

            print(f"Saved: {output_file}")

            # -------------------------------------------------
            # MEMORY INFORMATION
            # -------------------------------------------------

            print_gpu_memory()

        except Exception as e:

            print(
                f"\n[ERROR]"
                f"\nVideo: {video_name}"
                f"\nQuestion index: {idx}"
                f"\nQuestion: {question}"
                f"\nError: {repr(e)}"
            )

            # Save completed answers
            output_df.to_csv(
                output_file,
                index=False,
            )

            # Continue with next question
            continue

        finally:

            # ------------------------------------------------
            # RELEASE TENSORS
            # ------------------------------------------------

            if inputs is not None:
                del inputs

            if generated_ids is not None:
                del generated_ids

            if generated_ids_trimmed is not None:
                del generated_ids_trimmed

            if "messages" in locals():
                del messages

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
        index=False,
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

    process_video(video_path)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ALL VIDEOS COMPLETED")
print("=" * 70)

print_gpu_memory()