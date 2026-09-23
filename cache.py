import gc
import time
from pathlib import Path
import traceback
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

# Exactly 64 frames per video
NUM_FRAMES = 64

# Keep this relatively small for video QA
MAX_NEW_TOKENS = 512

QUESTION_COLUMN = "Questions"

# ------------------------------------------------------------
# IMPORTANT
#
# True  = keep cached visual features on CPU RAM.
#
# This is safer because your model is already using ~42 GB
# of the 48 GB GPU.
#
# False = keep visual features on GPU.
# This may be faster but can cause OOM.
# ------------------------------------------------------------

CACHE_ON_CPU = True


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# ------------------------------------------------------------------
# Qwen3-VL + Transformers 5.17 compatibility patch
# ------------------------------------------------------------------
_original_validate_model_kwargs = model._validate_model_kwargs

def _validate_model_kwargs_qwen3vl(model_kwargs):
    # Transformers 5.17's generation validator does not recognize
    # these Qwen3-VL-specific arguments, even though they are accepted
    # and forwarded by prepare_inputs_for_generation().
    check_kwargs = model_kwargs.copy()

    check_kwargs.pop("visual_pos_masks", None)
    check_kwargs.pop("deepstack_visual_embeds", None)

    return _original_validate_model_kwargs(check_kwargs)

model._validate_model_kwargs = _validate_model_kwargs_qwen3vl
processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Model loaded.")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )


# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted(
    VIDEO_DIR.glob("Video_ID_*.mp4"),
    key=lambda x: int(
        x.stem.split("_")[-1]
    )
)

print(
    f"Found {len(videos)} videos."
)


# ============================================================
# SAMPLE 64 FRAMES
# ============================================================

def sample_video_frames(
    video_path,
    num_frames=64
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
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            f"Could not determine frame count: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Uniformly sample across entire video
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
        desc="Sampling",
        leave=False,
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_idx)
        )

        success, frame = cap.read()

        if not success:

            print(
                f"[WARNING] Failed to read frame "
                f"{frame_idx}"
            )

            continue

        # BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = Image.fromarray(
            frame
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            "No frames extracted."
        )

    print(
        f"Sampled {len(frames)}/{num_frames} frames."
    )

    return frames


# ============================================================
# BUILD VISUAL CACHE
# ============================================================

def build_visual_cache(frames):

    print("\n" + "-" * 70)
    print("Building visual cache...")
    print("-" * 70)

    # --------------------------------------------------------
    # We need a multimodal prompt once so the processor
    # creates:
    #
    # pixel_values_videos
    # video_grid_thw
    #
    # The actual question doesn't matter for extracting
    # visual features.
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                },
                {
                    "type": "text",
                    "text": "Describe the video.",
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # Process the video
    # --------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Move video tensors to the model's input device
    # --------------------------------------------------------

    inputs = inputs.to(model.device)

    print(
        "Running Qwen vision encoder..."
    )

    t0 = time.time()

    with torch.inference_mode():

        video_outputs = model.get_video_features(
            pixel_values_videos=(
                inputs.pixel_values_videos
            ),
            video_grid_thw=(
                inputs.video_grid_thw
            ),
        )

    vision_time = time.time() - t0

    print(
        f"Vision encoding time: "
        f"{vision_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Qwen3-VL returns video embeddings split by video.
    #
    # We only have ONE video.
    # --------------------------------------------------------

    video_embeds = video_outputs.pooler_output

    video_embeds = torch.cat(
        video_embeds,
        dim=0
    )

    # --------------------------------------------------------
    # DeepStack features
    # --------------------------------------------------------

    deepstack_visual_embeds = (
        video_outputs.deepstack_features
    )

    # --------------------------------------------------------
    # Move cache to CPU RAM
    #
    # This is important because your GPU already uses
    # approximately 42 GB / 48 GB.
    # --------------------------------------------------------

    if CACHE_ON_CPU:

        video_embeds = (
            video_embeds
            .detach()
            .cpu()
        )

        deepstack_visual_embeds = [
            x.detach().cpu()
            for x in deepstack_visual_embeds
        ]

    else:

        video_embeds = (
            video_embeds
            .detach()
        )

        deepstack_visual_embeds = [
            x.detach()
            for x in deepstack_visual_embeds
        ]

    # --------------------------------------------------------
    # Save video grid information
    # --------------------------------------------------------

    video_grid_thw = (
        inputs.video_grid_thw
        .detach()
        .cpu()
    )

    print(
        f"Video embeddings shape: "
        f"{tuple(video_embeds.shape)}"
    )

    print(
        f"DeepStack levels: "
        f"{len(deepstack_visual_embeds)}"
    )

    print(
        "Visual features cached."
    )

    # --------------------------------------------------------
    # Free the temporary processor inputs
    # --------------------------------------------------------

    del inputs
    del video_outputs
    del messages

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return {
        "video_embeds": video_embeds,
        "deepstack_visual_embeds": (
            deepstack_visual_embeds
        ),
        "video_grid_thw": video_grid_thw,
    }


# ============================================================
# ANSWER ONE QUESTION USING CACHE
# ============================================================

def answer_question_with_cache(
    question,
    visual_cache,
    frames,
):

    # --------------------------------------------------------
    # We still give the processor the video frames here.
    #
    # IMPORTANT:
    #
    # This does NOT run the vision encoder.
    #
    # We only need the processor to construct:
    #
    # - video placeholder tokens
    # - attention mask
    # - position IDs
    # - video grid
    #
    # The actual visual features come from visual_cache.
    # --------------------------------------------------------

    messages = [
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
                        "using only the information available "
                        "in the video.\n\n"
                        f"Question: {question}\n\n"
                        "Give a concise and direct answer."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Move normal text/position inputs to model device
    # --------------------------------------------------------

    inputs = inputs.to(
        model.device
    )

    # --------------------------------------------------------
    # Get text embeddings
    # --------------------------------------------------------

    inputs_embeds = (
        model.get_input_embeddings()(
            inputs.input_ids
        )
    )

    # --------------------------------------------------------
    # Get cached visual embeddings
    # --------------------------------------------------------

    video_embeds = (
        visual_cache["video_embeds"]
    )

    deepstack_visual_embeds = (
        visual_cache[
            "deepstack_visual_embeds"
        ]
    )

    # --------------------------------------------------------
    # Move cache to same device/dtype as embeddings
    # --------------------------------------------------------

    video_embeds = video_embeds.to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )

    deepstack_visual_embeds = [
        x.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        for x in deepstack_visual_embeds
    ]

    # --------------------------------------------------------
    # Find video placeholder positions
    #
    # This is the same operation Qwen3-VL itself performs
    # internally when normal video features are supplied.
    # --------------------------------------------------------

    _, video_mask = (
        model.model.get_placeholder_mask(
            input_ids=inputs.input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
    )

    # Qwen's video mask is [batch, seq, 1]
    video_mask = video_mask[..., 0]

    visual_pos_masks = video_mask

    # --------------------------------------------------------
    # Insert cached video embeddings into text embeddings
    # --------------------------------------------------------

    inputs_embeds = inputs_embeds.clone()

    inputs_embeds = inputs_embeds.masked_scatter(
        video_mask.unsqueeze(-1),
        video_embeds,
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            #position_ids=inputs.position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=(
                deepstack_visual_embeds
            ),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    # --------------------------------------------------------
    # Remove prompt tokens
    # --------------------------------------------------------

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del inputs
    del inputs_embeds
    del generated_ids
    del generated_ids_trimmed
    del video_embeds
    del deepstack_visual_embeds
    del messages

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return answer


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
    # QUESTION CSV
    # ========================================================

    if not question_file.exists():

        print(
            f"[WARNING] Question file not found:"
            f"\n{question_file}"
        )

        return

    questions_df = pd.read_csv(
        question_file
    ).head(1)

    if QUESTION_COLUMN not in questions_df.columns:

        print(
            f"[ERROR] '{QUESTION_COLUMN}' "
            f"column not found."
        )

        print(
            f"Available columns: "
            f"{list(questions_df.columns)}"
        )

        return

    # ========================================================
    # OUTPUT
    # ========================================================

    if output_file.exists():

        output_df = pd.read_csv(
            output_file
        )

        # Only retain these columns
        if (
            "question" in output_df.columns
            and "answer" in output_df.columns
        ):

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

    else:

        output_df = pd.DataFrame({
            "question": questions_df[
                QUESTION_COLUMN
            ].astype(str),

            "answer": ""
        })

    # ========================================================
    # SAMPLE VIDEO
    # ========================================================

    try:

        frames = sample_video_frames(
            video_path,
            NUM_FRAMES
        )

    except Exception as e:

        print(
            f"[ERROR] Frame sampling failed:"
            f"\n{repr(e)}"
        )

        return

    # ========================================================
    # BUILD VISUAL CACHE ONCE
    # ========================================================

    try:

        visual_cache = build_visual_cache(
            frames
        )

    except Exception as e:

        print(
            f"[ERROR] Visual caching failed:"
            f"\n{repr(e)}"
        )

        del frames

        gc.collect()

        return

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
            output_df.loc[
                idx,
                "answer"
            ]
        )

        if (
            pd.notna(existing_answer)
            and str(existing_answer).strip() != ""
        ):

            continue

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
            f"Q: {question}"
        )

        # ----------------------------------------------------
        # Generate answer using cached visual features
        # ----------------------------------------------------

        t0 = time.time()

        try:

            answer = answer_question_with_cache(
                question=question,
                visual_cache=visual_cache,
                frames=frames,
            )

            elapsed = (
                time.time() - t0
            )

            print(
                f"Time: {elapsed:.2f}s"
            )

            print(
                f"Answer: {answer}"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_df.loc[
                idx,
                "answer"
            ] = answer

            output_df[
                ["question", "answer"]
            ].to_csv(
                output_file,
                index=False
            )

        except Exception as e:

            print(
                "\n[ERROR]"
            )

            print(
                f"Question: {question}"
            )

            print(
                f"Error: {repr(e)}"
            )
            traceback.print_exc()

            # Save whatever succeeded
            output_df[
                ["question", "answer"]
            ].to_csv(
                output_file,
                index=False
            )

            continue

    # ========================================================
    # FINAL SAVE
    # ========================================================

    output_df[
        ["question", "answer"]
    ].to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # CLEANUP VIDEO CACHE
    # ========================================================

    del visual_cache
    del frames
    del output_df
    del questions_df

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print("\n" + "-" * 70)

    print(
        f"Completed: {video_name}"
    )

    print(
        f"Output: {output_file}"
    )

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

for video_path in videos:

    process_video(
        video_path
    )


print("\n" + "=" * 70)
print("ALL VIDEOS COMPLETED")
print("=" * 70)