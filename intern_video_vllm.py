import gc
import time
from pathlib import Path

import pandas as pd
import torch

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "OpenGVLab/InternVideo2_5_Chat_8B"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"


# ============================================================
# VIDEO CONFIG
# ============================================================

# We want exactly 256 uniformly sampled frames.
NUM_FRAMES = 256


# ============================================================
# GENERATION CONFIG
# ============================================================

MAX_NEW_TOKENS = 512

QUESTION_COLUMN = "Questions"


# ============================================================
# MODEL CONFIG
# ============================================================

DTYPE = torch.bfloat16

DEVICE = "cuda"


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
print("InternVideo2.5-Chat-8B")
print("=" * 80)

print(f"Model              : {MODEL_NAME}")
print(f"Video directory    : {VIDEO_DIR}")
print(f"Question directory : {QUESTION_DIR}")
print(f"Output directory   : {OUTPUT_DIR}")
print(f"Frames per video   : {NUM_FRAMES}")
print(f"Max new tokens     : {MAX_NEW_TOKENS}")
print("=" * 80)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,
)

print("Tokenizer loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading InternVideo2.5-Chat-8B...")

t0 = time.time()

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map="auto",
)

model.eval()

t1 = time.time()

print(
    f"Model loaded in "
    f"{t1 - t0:.2f}s"
)


# ============================================================
# GENERATION CONFIG
# ============================================================

generation_config = dict(

    # Deterministic generation
    do_sample=False,

    temperature=0.0,

    # Maximum answer length
    max_new_tokens=MAX_NEW_TOKENS,

    top_p=0.1,

    num_beams=1,
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


        # ----------------------------------------------------
        # Validate output structure
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


            # ------------------------------------------------
            # Make sure row count matches
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
            # MODEL INPUT
            # =================================================

            #
            # IMPORTANT:
            #
            # We pass the COMPLETE video path.
            #
            # We do NOT extract frames here.
            #
            # InternVideo2.5's video loader reads the video
            # internally.
            #

            t0 = time.time()


            # =================================================
            # INTERNVIDEO2.5 CHAT
            # =================================================

            answer, chat_history = model.chat(

                tokenizer,

                # COMPLETE VIDEO
                video_path=str(video_path),

                # QUESTION
                user_prompt=question,

                # ------------------------------------------------
                # Exactly 256 frames
                #
                # The model's internal video loader samples
                # these frames from the complete video.
                # ------------------------------------------------
                max_num_frames=NUM_FRAMES,

                # ------------------------------------------------
                # Generation
                # ------------------------------------------------
                generation_config=generation_config,

                # ------------------------------------------------
                # Return history
                # ------------------------------------------------
                return_history=True,
            )


            t1 = time.time()


            # =================================================
            # CLEAN ANSWER
            # =================================================

            answer = str(
                answer
            ).strip()


            # =================================================
            # TIMING
            # =================================================

            total_time = (
                t1 - t0
            )


            print(
                f"Generation / Video processing: "
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