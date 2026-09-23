import gc
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd

from tqdm import tqdm
from PIL import Image
import time

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/gemma-4-12B-it"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

# Exactly 64 frames from every video
NUM_FRAMES = 64

# Maximum number of generated tokens
MAX_NEW_TOKENS = 512

QUESTION_COLUMN = "Questions"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Gemma 4 12B...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)

print("Model loaded.")


# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted(
    VIDEO_DIR.glob("*.mp4"),
    key=lambda x: x.name
)

print(f"Found {len(videos)} videos.")


# ============================================================
# SAMPLE VIDEO FRAMES
# ============================================================

# def sample_video_frames(video_path, num_frames=64):

#     print(f"\nSampling {num_frames} frames from:")
#     print(video_path)

#     cap = cv2.VideoCapture(str(video_path))

#     if not cap.isOpened():
#         raise RuntimeError(
#             f"Could not open video: {video_path}"
#         )

#     total_frames = int(
#         cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     )

#     if total_frames <= 0:
#         cap.release()
#         raise RuntimeError(
#             f"Could not determine frame count: {video_path}"
#         )

#     # --------------------------------------------------------
#     # Uniformly select frame indices
#     # --------------------------------------------------------

#     frame_indices = np.linspace(
#         0,
#         total_frames - 1,
#         num_frames,
#         dtype=np.int64,
#     )

#     frames = []

#     # --------------------------------------------------------
#     # Read selected frames
#     # --------------------------------------------------------

#     for frame_idx in tqdm(
#         frame_indices,
#         desc="Sampling frames",
#         leave=False,
#     ):

#         cap.set(
#             cv2.CAP_PROP_POS_FRAMES,
#             int(frame_idx)
#         )

#         success, frame = cap.read()

#         if not success:
#             print(
#                 f"[WARNING] Could not read frame "
#                 f"{frame_idx}"
#             )
#             continue

#         # OpenCV BGR -> RGB

#         frame = cv2.cvtColor(
#             frame,
#             cv2.COLOR_BGR2RGB
#         )

#         frame = Image.fromarray(frame)

#         frames.append(frame)

#     cap.release()

#     if len(frames) == 0:
#         raise RuntimeError(
#             f"No frames could be extracted from "
#             f"{video_path}"
#         )

#     print(
#         f"Successfully sampled "
#         f"{len(frames)}/{num_frames} frames."
#     )

#     return frames


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

        # Make sure output has correct columns

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

            # Keep only required columns

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
    # SAMPLE VIDEO ONCE
    # ========================================================

    # try:

    #     frames = sample_video_frames(
    #         video_path,
    #         NUM_FRAMES
    #     )

    # except Exception as e:

    #     print(
    #         f"[ERROR] Failed to sample video: "
    #         f"{repr(e)}"
    #     )

    #     return

    # print(
    #     f"Using {len(frames)} frames "
    #     f"for ALL questions in {video_name}"
    # )

    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for idx in tqdm(
        range(len(questions_df)),
        desc=f"Questions - {video_name}",
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
                            "video": str(video_path),
                        },

                        {
                            "type": "text",

                            "text": (
                                "Answer the following question "
                                "using the information "
                                "available in the video.\n\n"
                                f"Question: {question}\n\n"
                                "Give a concise and direct answer."
                            ),
                        },
                    ],
                }
            ]

            # =================================================
            # PROCESS INPUT
            # =================================================

            t0 = time.time()

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",

                # Disable reasoning for benchmarking
                enable_thinking=False,
            )

            t1 = time.time()

            # -------------------------------------------------
            # Move tensors to model device
            # -------------------------------------------------

            inputs = inputs.to(
                model.device
            )

            # =================================================
            # GENERATE
            # =================================================

            with torch.inference_mode():

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            t2 = time.time()

            print(
                f"Processor: {t1 - t0:.2f}s | "
                f"Generation: {t2 - t1:.2f}s | "
                f"Total: {t2 - t0:.2f}s"
            )

            # =================================================
            # REMOVE INPUT TOKENS
            # =================================================

            input_len = inputs["input_ids"].shape[-1]

            generated_ids_trimmed = (
                generated_ids[:, input_len:]
            )

            # =================================================
            # DECODE
            # =================================================

            response = processor.decode(
                generated_ids_trimmed[0],
                skip_special_tokens=True,
            )

            # Gemma 4 may return structured response
            # when parsing is supported.

            try:
                parsed_response = processor.parse_response(
                    response
                )

                if isinstance(parsed_response, str):
                    answer = parsed_response.strip()

                elif isinstance(parsed_response, dict):

                    # Prefer text/content if available

                    if "text" in parsed_response:
                        answer = str(
                            parsed_response["text"]
                        ).strip()

                    elif "content" in parsed_response:
                        answer = str(
                            parsed_response["content"]
                        ).strip()

                    else:
                        answer = str(
                            parsed_response
                        ).strip()

                else:
                    answer = str(
                        parsed_response
                    ).strip()

            except Exception:

                # Fallback to raw decoded response

                answer = response.strip()

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

            continue

        finally:

            # ------------------------------------------------
            # Release Gemma tensors
            # ------------------------------------------------

            if "inputs" in locals():
                del inputs

            if "generated_ids" in locals():
                del generated_ids

            if "generated_ids_trimmed" in locals():
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

    # ========================================================
    # RELEASE VIDEO FRAMES
    # ========================================================

    del frames

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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