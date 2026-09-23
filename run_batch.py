import gc
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
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

# ------------------------------------------------------------
# VIDEO
# ------------------------------------------------------------

NUM_FRAMES = 64

# ------------------------------------------------------------
# BATCHING
# ------------------------------------------------------------

# You have ~42 GB / 48 GB currently.
# Start with 2.
BATCH_SIZE = 2

# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------

MAX_NEW_TOKENS = 128

QUESTION_COLUMN = "Questions"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Loading Qwen3-VL-32B...")
print("=" * 70)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
)

# ------------------------------------------------------------
# VERY IMPORTANT FOR BATCH GENERATION
# Qwen documentation recommends left padding.
# ------------------------------------------------------------

processor.tokenizer.padding_side = "left"

model.eval()

print("Model loaded.")

if torch.cuda.is_available():

    print(
        f"CUDA device count: "
        f"{torch.cuda.device_count()}"
    )

    print(
        f"Current allocated VRAM: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    print(
        f"Current reserved VRAM: "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )


# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted(
    VIDEO_DIR.glob("*.mp4"),
    key=lambda x: x.name,
)

print(
    f"\nFound {len(videos)} videos."
)


# ============================================================
# SAMPLE VIDEO FRAMES
# ============================================================

def sample_video_frames(
    video_path,
    num_frames=64,
):

    print(
        f"\nSampling {num_frames} frames from:"
    )

    print(video_path)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            f"Could not determine frame count: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Uniform frame indices
    # --------------------------------------------------------

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=np.int64,
    )

    frames = []

    for frame_idx in tqdm(
        frame_indices,
        desc="Sampling frames",
        leave=False,
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_idx),
        )

        success, frame = cap.read()

        if not success:

            print(
                f"[WARNING] Could not read "
                f"frame {frame_idx}"
            )

            continue

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # NumPy → PIL
        # ----------------------------------------------------

        frame = Image.fromarray(
            frame
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            f"No frames could be extracted "
            f"from {video_path}"
        )

    print(
        f"Successfully sampled "
        f"{len(frames)}/{num_frames} frames."
    )

    return frames


# ============================================================
# BUILD MESSAGE
# ============================================================

def build_message(
    frames,
    question,
):

    return [
        {
            "role": "user",
            "content": [

                {
                    "type": "video",
                    "video": frames,
                },

                {
                    "type": "text",
                    "text": (
                        "Answer the following question "
                        "using only the information "
                        "available in the video.\n\n"
                        f"Question: {question}\n\n"
                        "Give a concise and direct answer."
                    ),
                },

            ],
        }
    ]


# ============================================================
# PROCESS ONE QUESTION BATCH
# ============================================================

def process_question_batch(
    frames,
    questions,
):
    """
    Process multiple questions about the SAME video
    in one generation call.

    Example:

        frames + Q1
        frames + Q2

    become one batch:

        batch[0] = frames + Q1
        batch[1] = frames + Q2
    """

    batch_size = len(questions)

    print(
        f"\nProcessing batch of "
        f"{batch_size} questions..."
    )

    # --------------------------------------------------------
    # Build batch messages
    # --------------------------------------------------------

    messages_batch = []

    for question in questions:

        messages = build_message(
            frames,
            question,
        )

        messages_batch.append(
            messages
        )

    # --------------------------------------------------------
    # PROCESS INPUTS
    # --------------------------------------------------------

    processor_start = time.perf_counter()

    inputs = processor.apply_chat_template(
        messages_batch,

        tokenize=True,

        add_generation_prompt=True,

        return_dict=True,

        return_tensors="pt",

        padding=True,
    )

    processor_time = (
        time.perf_counter()
        - processor_start
    )

    # --------------------------------------------------------
    # Move tensors to model device
    # --------------------------------------------------------

    inputs = inputs.to(
        model.device
    )

    # --------------------------------------------------------
    # Report input information
    # --------------------------------------------------------

    if "input_ids" in inputs:

        input_lengths = (
            inputs.attention_mask.sum(
                dim=1
            )
            .tolist()
        )

        print(
            "Input token lengths:",
            input_lengths,
        )

    # --------------------------------------------------------
    # VRAM before generation
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        vram_before = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        torch.cuda.reset_peak_memory_stats()

    else:

        vram_before = 0

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    generation_start = time.perf_counter()

    with torch.inference_mode():

        generated_ids = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=True,

        )

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    # --------------------------------------------------------
    # Peak VRAM
    # --------------------------------------------------------

    if torch.cuda.is_available():

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

    else:

        peak_vram = 0

    # --------------------------------------------------------
    # REMOVE INPUT TOKENS
    #
    # IMPORTANT:
    #
    # Because we use LEFT padding, simply doing:
    #
    # generated_ids[len(input_ids):]
    #
    # is not safe for a batch.
    #
    # Use attention_mask to determine the actual
    # number of input tokens for each example.
    # --------------------------------------------------------

    generated_ids_trimmed = []

    for i in range(
        generated_ids.shape[0]
    ):

        input_length = int(
            inputs.attention_mask[
                i
            ].sum().item()
        )

        output_ids = generated_ids[
            i,
            input_length:,
        ]

        generated_ids_trimmed.append(
            output_ids
        )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    answers = processor.batch_decode(
        generated_ids_trimmed,

        skip_special_tokens=True,

        clean_up_tokenization_spaces=False,
    )

    answers = [
        answer.strip()
        for answer in answers
    ]

    total_time = (
        processor_time
        + generation_time
    )

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    print(
        f"\nProcessor time: "
        f"{processor_time:.2f}s"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f}s"
    )

    print(
        f"Total batch time: "
        f"{total_time:.2f}s"
    )

    print(
        f"Average/question: "
        f"{total_time / batch_size:.2f}s"
    )

    if torch.cuda.is_available():

        print(
            f"VRAM before generation: "
            f"{vram_before:.2f} GB"
        )

        print(
            f"Peak VRAM: "
            f"{peak_vram:.2f} GB"
        )

    print(
        f"Throughput: "
        f"{batch_size / total_time:.3f} "
        f"questions/sec"
    )

    # --------------------------------------------------------
    # CLEAN TEMPORARY TENSORS
    # --------------------------------------------------------

    del inputs
    del generated_ids
    del generated_ids_trimmed

    return (
        answers,
        processor_time,
        generation_time,
        peak_vram,
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    video_name = video_path.stem

    question_file = (
        QUESTION_DIR
        / f"{video_name}.csv"
    )

    output_file = (
        OUTPUT_DIR
        / f"{video_name}_answers.csv"
    )

    print("\n" + "=" * 70)

    print(
        f"VIDEO: {video_name}"
    )

    print("=" * 70)

    # ========================================================
    # CHECK QUESTION FILE
    # ========================================================

    if not question_file.exists():

        print(
            f"[WARNING] Question file not found:"
            f"\n{question_file}"
        )

        return

    questions_df = pd.read_csv(
        question_file
    )

    if (
        QUESTION_COLUMN
        not in questions_df.columns
    ):

        print(
            f"[ERROR] Column "
            f"'{QUESTION_COLUMN}' "
            f"not found."
        )

        print(
            "Available columns:",
            list(questions_df.columns),
        )

        return

    # ========================================================
    # LOAD / CREATE OUTPUT
    # ========================================================

    if output_file.exists():

        output_df = pd.read_csv(
            output_file
        )

        print(
            f"Existing output found:"
            f" {len(output_df)} rows"
        )

        # ----------------------------------------------------
        # Make sure output has correct structure
        # ----------------------------------------------------

        if (
            "question"
            not in output_df.columns
            or
            "answer"
            not in output_df.columns
        ):

            print(
                "[WARNING] Existing output "
                "has incorrect columns."
            )

            output_df = pd.DataFrame({
                "question": (
                    questions_df[
                        QUESTION_COLUMN
                    ].astype(str)
                ),

                "answer": "",
            })

        else:

            output_df = output_df[
                [
                    "question",
                    "answer",
                ]
            ]

    else:

        output_df = pd.DataFrame({

            "question": (
                questions_df[
                    QUESTION_COLUMN
                ].astype(str)
            ),

            "answer": "",
        })

    # ========================================================
    # SAMPLE VIDEO ONCE
    # ========================================================

    try:

        frames = sample_video_frames(
            video_path,
            NUM_FRAMES,
        )

    except Exception as e:

        print(
            f"[ERROR] Failed to sample video:"
            f"\n{repr(e)}"
        )

        return

    print(
        f"\nUsing {len(frames)} frames "
        f"for ALL questions."
    )

    # ========================================================
    # FIND PENDING QUESTIONS
    # ========================================================

    pending_indices = []

    for idx in range(
        len(questions_df)
    ):

        existing_answer = (
            output_df.loc[
                idx,
                "answer",
            ]
        )

        if (
            pd.notna(existing_answer)
            and
            str(existing_answer).strip()
            != ""
        ):

            continue

        pending_indices.append(
            idx
        )

    if len(pending_indices) == 0:

        print(
            "All questions already "
            "have answers."
        )

        del frames

        return

    print(
        f"Pending questions: "
        f"{len(pending_indices)}"
    )

    # ========================================================
    # BATCH QUESTIONS
    # ========================================================

    for batch_start in range(
        0,
        len(pending_indices),
        BATCH_SIZE,
    ):

        batch_indices = (
            pending_indices[
                batch_start:
                batch_start + BATCH_SIZE
            ]
        )

        questions = [

            str(
                questions_df.loc[
                    idx,
                    QUESTION_COLUMN,
                ]
            )

            for idx in batch_indices
        ]

        print("\n" + "-" * 70)

        print(
            f"Batch "
            f"{batch_start // BATCH_SIZE + 1}"
            f"/"
            f"{(
                len(pending_indices)
                + BATCH_SIZE
                - 1
            ) // BATCH_SIZE}"
        )

        for i, question in enumerate(
            questions
        ):

            print(
                f"  Q{i + 1}: "
                f"{question}"
            )

        # ====================================================
        # PROCESS BATCH
        # ====================================================

        try:

            (
                answers,
                processor_time,
                generation_time,
                peak_vram,
            ) = process_question_batch(
                frames,
                questions,
            )

            # =================================================
            # STORE ANSWERS
            # =================================================

            for idx, answer in zip(
                batch_indices,
                answers,
            ):

                output_df.loc[
                    idx,
                    "answer",
                ] = answer

            # =================================================
            # SAVE AFTER EACH BATCH
            # =================================================

            output_df[
                [
                    "question",
                    "answer",
                ]
            ].to_csv(
                output_file,
                index=False,
            )

            # =================================================
            # PRINT ANSWERS
            # =================================================

            for question, answer in zip(
                questions,
                answers,
            ):

                print(
                    "\nQuestion:",
                    question,
                )

                print(
                    "Answer:",
                    answer,
                )

        except torch.cuda.OutOfMemoryError:

            print(
                "\n" + "!" * 70
            )

            print(
                "CUDA OUT OF MEMORY"
            )

            print(
                "!" * 70
            )

            print(
                f"Batch size {BATCH_SIZE} "
                f"is too large for this configuration."
            )

            print(
                "Try:"
            )

            print(
                "BATCH_SIZE = 1"
            )

            print(
                "or reduce NUM_FRAMES / video pixel budget."
            )

            # ------------------------------------------------
            # Clear memory
            # ------------------------------------------------

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            # ------------------------------------------------
            # Stop this video rather than corrupting results
            # ------------------------------------------------

            return

        except Exception as e:

            print(
                "\n[ERROR]"
            )

            print(
                f"Video: {video_name}"
            )

            print(
                f"Batch indices: "
                f"{batch_indices}"
            )

            print(
                f"Error: {repr(e)}"
            )

            # ------------------------------------------------
            # Save whatever was completed
            # ------------------------------------------------

            output_df[
                [
                    "question",
                    "answer",
                ]
            ].to_csv(
                output_file,
                index=False,
            )

            continue

        finally:

            # ------------------------------------------------
            # DO NOT call empty_cache() here.
            #
            # We're processing another batch and PyTorch's
            # allocator can reuse the memory.
            # ------------------------------------------------

            gc.collect()

    # ========================================================
    # FINAL SAVE
    # ========================================================

    output_df[
        [
            "question",
            "answer",
        ]
    ].to_csv(
        output_file,
        index=False,
    )

    # ========================================================
    # RELEASE VIDEO
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

    process_video(
        video_path
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ALL VIDEOS COMPLETED")
print("=" * 70)