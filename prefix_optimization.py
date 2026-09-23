import gc
import copy
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
    DynamicCache,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

DATASET_DIR = Path("dataset")

VIDEO_DIR = DATASET_DIR / "videos"
QUESTION_DIR = DATASET_DIR / "questions"
OUTPUT_DIR = DATASET_DIR / "output"

NUM_FRAMES = 64

# Keep this relatively small for QA.
MAX_NEW_TOKENS = 128

QUESTION_COLUMN = "Questions"

# Qwen3-VL visual resolution.
# Start conservative and benchmark upward.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Qwen3-VL-32B...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
)

# Important for visual processing.
# Qwen3-VL uses patch size 16 in qwen-vl-utils.
try:
    processor.image_processor.size["shortest_edge"] = MIN_PIXELS
    processor.image_processor.size["longest_edge"] = MAX_PIXELS
except Exception:
    pass

try:
    processor.video_processor.size["shortest_edge"] = MIN_PIXELS
    processor.video_processor.size["longest_edge"] = MAX_PIXELS
except Exception:
    pass

model.eval()

print("Model loaded.")
print(f"Model device: {model.device}")


# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted(
    VIDEO_DIR.glob("*.mp4"),
    key=lambda x: x.name,
)

print(f"Found {len(videos)} videos.")


# ============================================================
# SAMPLE VIDEO FRAMES
# ============================================================

def sample_video_frames(video_path, num_frames=64):

    print(f"\nSampling {num_frames} frames from:")
    print(video_path)

    cap = cv2.VideoCapture(str(video_path))

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
            f"Could not determine frame count: {video_path}"
        )

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
                f"[WARNING] Could not read frame "
                f"{frame_idx}"
            )
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = Image.fromarray(frame)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames could be extracted from "
            f"{video_path}"
        )

    print(
        f"Successfully sampled "
        f"{len(frames)}/{num_frames} frames."
    )

    return frames


# ============================================================
# BUILD MESSAGE
# ============================================================

def build_messages(frames, question):

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
# MOVE INPUTS TO MODEL DEVICE
# ============================================================

def move_inputs_to_device(inputs):

    result = {}

    for key, value in inputs.items():

        if isinstance(value, torch.Tensor):
            result[key] = value.to(
                model.device,
                non_blocking=True,
            )

        else:
            result[key] = value

    return result


# ============================================================
# FIND COMMON PREFIX
# ============================================================

def find_common_prefix_length(input_ids_list):
    """
    Find the number of tokens shared by all question prompts.

    All prompts have the same:

        <video>
        fixed instruction
        "Question:"

    and differ only after that.
    """

    if len(input_ids_list) == 0:
        raise ValueError(
            "No input IDs supplied."
        )

    first = input_ids_list[0][0]

    max_prefix = first.shape[0]

    for ids in input_ids_list[1:]:

        current = ids[0]

        limit = min(
            max_prefix,
            current.shape[0],
        )

        mismatch = (
            first[:limit] != current[:limit]
        )

        mismatch_indices = torch.nonzero(
            mismatch,
            as_tuple=False,
        )

        if mismatch_indices.numel() > 0:

            max_prefix = min(
                max_prefix,
                int(mismatch_indices[0].item()),
            )

    return max_prefix


# ============================================================
# PREPARE ALL QUESTION INPUTS
# ============================================================

def prepare_question_inputs(frames, questions):

    """
    IMPORTANT:

    We process the actual video only ONCE.

    The first question creates the complete
    multimodal processor output.

    The remaining question prompts are created
    from the same multimodal template.

    """

    if len(questions) == 0:
        return [], None, None

    print(
        "\nPreparing multimodal video input..."
    )

    # --------------------------------------------------------
    # First question
    # --------------------------------------------------------

    first_messages = build_messages(
        frames,
        questions[0],
    )

    first_inputs = processor.apply_chat_template(
        first_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    first_inputs = move_inputs_to_device(
        first_inputs
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The actual visual tensors from the first call
    # are reused.
    # --------------------------------------------------------

    visual_keys = [
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "video_metadata",
        "mm_token_type_ids",
    ]

    shared_visual_inputs = {}

    for key in visual_keys:

        if key in first_inputs:
            shared_visual_inputs[key] = (
                first_inputs[key]
            )

    # --------------------------------------------------------
    # We need the text template for every question.
    #
    # tokenize=False does NOT perform the expensive
    # visual tensor processing.
    # --------------------------------------------------------

    text_prompts = []

    for question in questions:

        messages = build_messages(
            frames,
            question,
        )

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        text_prompts.append(text)

    # --------------------------------------------------------
    # Tokenize the text prompts.
    #
    # We deliberately do NOT pass video tensors here.
    # --------------------------------------------------------

    tokenizer = processor.tokenizer

    text_inputs = tokenizer(
        text_prompts,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )

    text_input_ids = text_inputs["input_ids"]

    # --------------------------------------------------------
    # Because the video placeholder is expanded by the
    # multimodal processor, the first processed input has
    # more tokens than the raw text tokenizer output.
    #
    # Therefore we build each full input by replacing the
    # single video placeholder region using the first
    # processed input.
    # --------------------------------------------------------

    processed_reference_ids = (
        first_inputs["input_ids"]
    )

    video_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|video_pad|>"
    )

    reference_video_positions = (
        processed_reference_ids[0]
        == video_token_id
    )

    video_positions = torch.nonzero(
        reference_video_positions,
        as_tuple=False,
    ).flatten()

    if video_positions.numel() == 0:

        raise RuntimeError(
            "Could not locate expanded video tokens "
            "in the processed Qwen3-VL input."
        )

    video_start = int(
        video_positions[0].item()
    )

    video_end = int(
        video_positions[-1].item()
    ) + 1

    # --------------------------------------------------------
    # Find the raw video placeholder in the tokenized
    # text prompt.
    # --------------------------------------------------------

    raw_video_token_id = video_token_id

    raw_video_positions = (
        text_input_ids[0][0]
        == raw_video_token_id
    )

    raw_video_indices = torch.nonzero(
        raw_video_positions,
        as_tuple=False,
    ).flatten()

    if raw_video_indices.numel() == 0:

        raise RuntimeError(
            "Could not find raw <|video_pad|> "
            "placeholder."
        )

    raw_video_index = int(
        raw_video_indices[0].item()
    )

    # --------------------------------------------------------
    # Build full token sequences.
    # --------------------------------------------------------

    full_input_ids_list = []

    for i, ids in enumerate(
        text_input_ids
    ):

        ids = ids.to(model.device)

        before_video = ids[
            :raw_video_index
        ]

        after_video = ids[
            raw_video_index + 1:
        ]

        expanded_video = (
            processed_reference_ids[0][
                video_start:video_end
            ]
        )

        reconstructed = torch.cat(
            [
                before_video,
                expanded_video,
                after_video,
            ],
            dim=0,
        )

        full_input_ids_list.append(
            reconstructed.unsqueeze(0)
        )

    # --------------------------------------------------------
    # Find common prefix.
    # --------------------------------------------------------

    prefix_len = find_common_prefix_length(
        full_input_ids_list
    )

    print(
        f"Total tokens for first question: "
        f"{full_input_ids_list[0].shape[1]}"
    )

    print(
        f"Shared prefix tokens: "
        f"{prefix_len}"
    )

    print(
        f"Question-specific tokens: "
        f"{full_input_ids_list[0].shape[1] - prefix_len}"
    )

    return (
        full_input_ids_list,
        shared_visual_inputs,
        first_inputs,
        prefix_len,
    )


# ============================================================
# CREATE PREFIX CACHE
# ============================================================

def create_video_prefix_cache(
    reference_inputs,
    prefix_len,
):
    """
    Run the shared video-containing prefix once.

    This is the expensive multimodal prefill.
    """

    print("\nCreating video prefix KV cache...")

    prefix_inputs = {}

    # --------------------------------------------------------
    # input_ids
    # --------------------------------------------------------

    prefix_inputs["input_ids"] = (
        reference_inputs["input_ids"][
            :, :prefix_len
        ]
    )

    # --------------------------------------------------------
    # attention mask
    # --------------------------------------------------------

    if "attention_mask" in reference_inputs:

        prefix_inputs["attention_mask"] = (
            reference_inputs["attention_mask"][
                :, :prefix_len
            ]
        )

    # --------------------------------------------------------
    # position IDs
    # --------------------------------------------------------

    if "position_ids" in reference_inputs:

        position_ids = (
            reference_inputs["position_ids"]
        )

        if position_ids.ndim == 3:

            prefix_inputs["position_ids"] = (
                position_ids[
                    :, :, :prefix_len
                ]
            )

        else:

            prefix_inputs["position_ids"] = (
                position_ids[
                    :, :prefix_len
                ]
            )

    # --------------------------------------------------------
    # Multimodal inputs
    # --------------------------------------------------------

    for key in [
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "mm_token_type_ids",
    ]:

        if key not in reference_inputs:
            continue

        value = reference_inputs[key]

        if key == "mm_token_type_ids":

            prefix_inputs[key] = value[
                :, :prefix_len
            ]

        else:

            # Visual tensors are NOT sequence tensors.
            # Keep them intact.
            prefix_inputs[key] = value

    # --------------------------------------------------------
    # Dynamic KV cache
    # --------------------------------------------------------

    cache = DynamicCache(
        config=model.config
    )

    start = time.time()

    with torch.inference_mode():

        outputs = model(
            **prefix_inputs,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    elapsed = time.time() - start

    cache = outputs.past_key_values

    print(
        f"Prefix prefill time: {elapsed:.2f}s"
    )

    print(
        f"Cached sequence length: "
        f"{cache.get_seq_length()}"
    )

    return cache


# ============================================================
# GENERATE USING PREFIX CACHE
# ============================================================

def generate_with_prefix_cache(
    full_input_ids,
    prefix_cache,
    prefix_len,
    reference_inputs,
):
    """
    Use the cached video prefix and process only
    the question-specific suffix.
    """

    # --------------------------------------------------------
    # Clone the cache so this question doesn't destroy
    # the reusable prefix cache.
    # --------------------------------------------------------

    cache = copy.deepcopy(
        prefix_cache
    )

    full_ids = full_input_ids.to(
        model.device
    )

    # --------------------------------------------------------
    # Question-specific portion
    # --------------------------------------------------------

    suffix_ids = full_ids[
        :, prefix_len:
    ]

    if suffix_ids.shape[1] == 0:
        raise RuntimeError(
            "Question suffix is empty."
        )

    # --------------------------------------------------------
    # Attention mask must describe:
    #
    # cached prefix + new suffix
    #
    # --------------------------------------------------------

    full_attention_mask = torch.ones(
        (
            1,
            full_ids.shape[1],
        ),
        dtype=torch.long,
        device=model.device,
    )

    # --------------------------------------------------------
    # Position IDs
    #
    # Use the positions from the original multimodal
    # prompt for the suffix.
    # --------------------------------------------------------

    suffix_position_ids = None

    if "position_ids" in reference_inputs:

        reference_position_ids = (
            reference_inputs[
                "position_ids"
            ]
        )

        if reference_position_ids.ndim == 3:

            suffix_position_ids = (
                reference_position_ids[
                    :,
                    :,
                    prefix_len:
                ]
            )

        else:

            suffix_position_ids = (
                reference_position_ids[
                    :,
                    prefix_len:
                ]
            )

    # --------------------------------------------------------
    # cache_position
    # --------------------------------------------------------

    cache_position = torch.arange(
        prefix_len,
        full_ids.shape[1],
        device=model.device,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Question prefill
    # --------------------------------------------------------

    start = time.time()

    with torch.inference_mode():

        outputs = model(
            input_ids=suffix_ids,
            attention_mask=full_attention_mask,
            position_ids=suffix_position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    prefill_time = time.time() - start

    # --------------------------------------------------------
    # First generated token
    # --------------------------------------------------------

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    generated_tokens = []

    # --------------------------------------------------------
    # Decode loop
    # --------------------------------------------------------

    generation_start = time.time()

    for step in range(MAX_NEW_TOKENS):

        token_id = int(
            next_token.item()
        )

        # EOS
        if (
            model.generation_config.eos_token_id
            is not None
            and token_id
            == model.generation_config.eos_token_id
        ):
            break

        generated_tokens.append(
            token_id
        )

        # --------------------------------------------
        # Position for the newly generated token
        # --------------------------------------------

        if suffix_position_ids is not None:

            last_position = (
                suffix_position_ids[
                    ...,
                    -1:
                ]
            )

            next_position_ids = (
                last_position + step + 1
            )

        else:

            next_position_ids = None

        # --------------------------------------------
        # New cache position
        # --------------------------------------------

        new_cache_position = torch.tensor(
            [
                prefix_len
                + suffix_ids.shape[1]
                + step
            ],
            device=model.device,
            dtype=torch.long,
        )

        # --------------------------------------------
        # Forward one token
        # --------------------------------------------

        with torch.inference_mode():

            outputs = model(
                input_ids=next_token,
                attention_mask=torch.ones(
                    (
                        1,
                        prefix_len
                        + suffix_ids.shape[1]
                        + step
                        + 1,
                    ),
                    dtype=torch.long,
                    device=model.device,
                ),
                position_ids=next_position_ids,
                cache_position=new_cache_position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

    generation_time = (
        time.time() - generation_start
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    if len(generated_tokens) == 0:
        answer = ""
    else:

        generated_tensor = torch.tensor(
            generated_tokens,
            dtype=torch.long,
            device=model.device,
        ).unsqueeze(0)

        answer = processor.batch_decode(
            generated_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return (
        answer,
        prefill_time,
        generation_time,
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
    print(f"VIDEO: {video_name}")
    print("=" * 70)

    # ========================================================
    # QUESTION CSV
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

        return

    # ========================================================
    # OUTPUT
    # ========================================================

    if output_file.exists():

        output_df = pd.read_csv(
            output_file
        )

        if (
            "question" not in output_df.columns
            or "answer" not in output_df.columns
        ):

            output_df = pd.DataFrame({
                "question": questions_df[
                    QUESTION_COLUMN
                ].astype(str),
                "answer": "",
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
            "answer": "",
        })

    # ========================================================
    # SAMPLE VIDEO
    # ========================================================

    try:

        frames = sample_video_frames(
            video_path,
            NUM_FRAMES,
        )

    except Exception as e:

        print(
            f"[ERROR] Failed to sample video: "
            f"{repr(e)}"
        )

        return

    print(
        f"Using {len(frames)} frames "
        f"for all questions."
    )

    # ========================================================
    # GET QUESTIONS THAT STILL NEED ANSWERS
    # ========================================================

    pending_indices = []

    for idx in range(
        len(questions_df)
    ):

        existing_answer = (
            output_df.loc[
                idx,
                "answer"
            ]
        )

        if (
            pd.notna(existing_answer)
            and str(existing_answer).strip()
            != ""
        ):
            continue

        pending_indices.append(idx)

    if len(pending_indices) == 0:

        print(
            "All questions already answered."
        )

        return

    questions = [
        str(
            questions_df.loc[
                idx,
                QUESTION_COLUMN
            ]
        )
        for idx in pending_indices
    ]

    print(
        f"Pending questions: "
        f"{len(questions)}"
    )

    # ========================================================
    # PREPARE QUESTIONS
    # ========================================================

    try:

        (
            full_input_ids_list,
            shared_visual_inputs,
            reference_inputs,
            prefix_len,
        ) = prepare_question_inputs(
            frames,
            questions,
        )

    except Exception as e:

        print(
            "\n[ERROR] Failed to prepare "
            f"video inputs:\n{repr(e)}"
        )

        return

    # ========================================================
    # CREATE PREFIX CACHE
    # ========================================================

    try:

        prefix_cache = (
            create_video_prefix_cache(
                reference_inputs,
                prefix_len,
            )
        )

    except Exception as e:

        print(
            "\n[ERROR] Failed to create "
            f"prefix cache:\n{repr(e)}"
        )

        return

    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for local_idx, (
        original_idx,
        question,
        full_input_ids,
    ) in enumerate(
        zip(
            pending_indices,
            questions,
            full_input_ids_list,
        )
    ):

        print("\n" + "-" * 70)

        print(
            f"Question "
            f"{local_idx + 1}/"
            f"{len(questions)}"
        )

        print(
            f"{question}"
        )

        try:

            answer, prefill_time, generation_time = (
                generate_with_prefix_cache(
                    full_input_ids,
                    prefix_cache,
                    prefix_len,
                    reference_inputs,
                )
            )

            total_time = (
                prefill_time
                + generation_time
            )

            print(
                f"Prefix/question prefill: "
                f"{prefill_time:.2f}s"
            )

            print(
                f"Generation: "
                f"{generation_time:.2f}s"
            )

            print(
                f"Total: "
                f"{total_time:.2f}s"
            )

            print(
                f"Answer: {answer}"
            )

            output_df.loc[
                original_idx,
                "answer",
            ] = answer

            # ------------------------------------------------
            # Save periodically
            # ------------------------------------------------

            if (
                (local_idx + 1) % 5 == 0
                or local_idx
                == len(questions) - 1
            ):

                output_df[
                    ["question", "answer"]
                ].to_csv(
                    output_file,
                    index=False,
                )

        except Exception as e:

            print(
                "\n[ERROR]"
                f"\nVideo: {video_name}"
                f"\nQuestion index: "
                f"{original_idx}"
                f"\nQuestion: {question}"
                f"\nError: {repr(e)}"
            )

            output_df[
                ["question", "answer"]
            ].to_csv(
                output_file,
                index=False,
            )

        finally:

            # DO NOT empty CUDA cache after every question.
            #
            # The prefix cache is intentionally being retained.
            #
            gc.collect()

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
    # RELEASE
    # ========================================================

    del prefix_cache
    del reference_inputs
    del full_input_ids_list
    del shared_visual_inputs
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
# MAIN
# ============================================================

for video_path in videos:

    process_video(video_path)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ALL VIDEOS COMPLETED")
print("=" * 70)