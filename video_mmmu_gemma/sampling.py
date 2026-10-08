"""
Run Gemma on all Video-MMMU videos; save one CSV per video (Video_ID_1.csv ...),
each with that video's Perception / Comprehension / Adaptation rows.

1) Set the paths in the CONFIG block below (or override with command-line flags).
2) python eval_videommmu_300.py
Finished videos are skipped, so you can stop and resume. Test first with LIMIT_VIDEOS = 3.
"""
# ============================ CONFIG: EDIT THESE ============================
QUESTIONS_DIR = "/workspace/gurrt/VideoMMMU/data/Questions"          # folder with Video_ID_1.csv ... Video_ID_300.csv
VIDEO_DIR     = "/workspace/gurrt/VideoMMMU/data/Videos"             # folder with the video files (searched recursively)
IMAGE_DIR     = "/workspace/gurrt/VideoMMMU/data/dataset/images"                            # folder with Adaptation images; "" 
OUT_DIR       = "/workspace/gurrt/videommmu-gemma-12B/result"            # per-video CSVs are written here
MODEL_ID      = "google/gemma-4-12B-it"       # confirm the exact ID on the HF model card
NUM_FRAMES    = 32                            # frames sampled per video (paper used 32 or 64 for open models)
MAX_SIDE      = 768                           # longest image side in pixels
MAX_NEW_TOKENS = 512                          # questions ask the model to reason, then give "Answer: X"
LIMIT_VIDEOS  = 3                            # 0 = all 300; set e.g. 3 for a test run
# ============================================================================

import argparse, os, re
import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModel
except ImportError:
    from transformers import AutoModelForMultimodalLM as AutoModel

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def build_index(root, exts):
    idx = {}
    for dp, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in exts:
                idx.setdefault(stem, os.path.join(dp, f))
    return idx


def sample_frames(path, n, max_side):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / max(n, 1)))
        ok, f = cap.read()
        if ok:
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            img.thumbnail((max_side, max_side))
            frames.append(img)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def extract_pred(text, is_mc):
    """Take the LAST 'Answer: X' line. No match -> empty (counted wrong)."""
    if is_mc:
        m = re.findall(r"Answer\s*[:：]\s*\**\(?([A-J])\b", text, flags=re.I)
        return m[-1].upper() if m else ""
    m = re.findall(r"Answer\s*[:：]\s*(.+)", text, flags=re.I)
    return m[-1].strip() if m else ""


def is_correct(pred, gt, is_mc):
    if is_mc:
        return pred == str(gt).strip().upper()
    try:
        return abs(float(pred.replace(",", "")) - float(str(gt).replace(",", ""))) <= 0.01 * abs(float(str(gt)))
    except ValueError:
        return pred.strip().lower() == str(gt).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", default=QUESTIONS_DIR)
    ap.add_argument("--video_dir", default=VIDEO_DIR)
    ap.add_argument("--image_dir", default=IMAGE_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--limit_videos", type=int, default=LIMIT_VIDEOS)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if os.path.abspath(a.questions_dir) == os.path.abspath(a.out_dir):
        raise SystemExit("OUT_DIR must be a different folder from QUESTIONS_DIR (file names would clash).")
    qfiles = [f for f in os.listdir(a.questions_dir) if f.lower().endswith(".csv")]
    qfiles.sort(key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])  # 1,2,...,10,...
    print(f"{len(qfiles)} question files found in {a.questions_dir}")

    vid_index = build_index(a.video_dir, VIDEO_EXT)
    img_index = build_index(a.image_dir or a.video_dir, IMAGE_EXT)
    if not a.image_dir:  # also look next to the video folder (e.g. .../dataset/images)
        img_index.update({k: v for k, v in build_index(os.path.dirname(a.video_dir.rstrip("/")), IMAGE_EXT).items()
                          if k not in img_index})
    print(f"indexed {len(vid_index)} video files, {len(img_index)} image files")

    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()

    order = {"Perception": 0, "Comprehension": 1, "Adaptation": 2}
    if a.limit_videos:
        qfiles = qfiles[: a.limit_videos]
    names = qfiles

    for n, qfile in enumerate(qfiles, 1):
        vname = os.path.splitext(qfile)[0]          # e.g. Video_ID_1
        out_path = os.path.join(a.out_dir, f"{vname}.csv")
        if os.path.exists(out_path):
            continue
        g = pd.read_csv(os.path.join(a.questions_dir, qfile))
        g["_o"] = g["track"].map(order)
        g = g.sort_values("_o").drop(columns="_o")
        vid_id = g.iloc[0]["video_id"]

        frames, err = [], ""
        try:
            vpath = vid_index.get(str(vid_id)) or vid_index.get(str(vname))
            if not vpath:
                raise FileNotFoundError(f"no video file named {vid_id} or {vname} under {a.video_dir}")
            frames = sample_frames(vpath, a.frames, MAX_SIDE)
        except Exception as e:
            err = f"ERROR(video): {e}"

        rows = []
        for _, r in g.iterrows():
            is_mc = str(r["question_type"]).startswith("multiple")
            resp = err
            if not err:
                try:
                    content = [{"type": "image", "image": f} for f in frames]
                    prompt = str(r["question"])
                    ipath = r.get("image_file")
                    if isinstance(ipath, str) and ipath.strip():
                        if not os.path.isfile(ipath):  # path from another machine: find by file name
                            ipath = img_index.get(os.path.splitext(os.path.basename(ipath))[0])
                        if not ipath:
                            raise FileNotFoundError(f"image for {r['question_id']} not found")
                        content.append({"type": "image", "image": Image.open(ipath).convert("RGB")})
                        prompt = "The image for this question is the final frame of the video.\n" + prompt
                    content.append({"type": "text", "text": prompt})
                    inputs = processor.apply_chat_template(
                        [{"role": "user", "content": content}], add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)
                    with torch.inference_mode():
                        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                except Exception as e:
                    resp = f"ERROR: {e}"
            pred = extract_pred(resp, is_mc)
            rows.append({
                "video_name": vname, "video_id": vid_id, "question_id": r["question_id"],
                "track": r["track"], "qa_type": r["qa_type"], "subject": r["subject"],
                "subject_group": r["subject_group"], "gt_answer": r["gt_answer"],
                "pred": pred, "correct": int(is_correct(pred, r["gt_answer"], is_mc)), "response": resp,
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[{n}/{len(names)}] {vname} ({vid_id}): " +
              ", ".join(f"{x['track'][:4]}={x['correct']}" for x in rows), flush=True)

    parts = [pd.read_csv(os.path.join(a.out_dir, f)) for f in sorted(os.listdir(a.out_dir))
             if f.endswith(".csv") and not f.startswith("_")]
    allr = pd.concat(parts, ignore_index=True)
    allr.to_csv(os.path.join(a.out_dir, "_all_results.csv"), index=False)
    print(f"\nVideos: {allr['video_name'].nunique()}   Questions: {len(allr)}")
    print(f"Overall accuracy: {100 * allr['correct'].mean():.2f}%")
    print((100 * allr.groupby("track")["correct"].mean()).round(2).to_string())
    bad = (allr["pred"].isna() | (allr["pred"].astype(str) == "")).sum()
    print(f"Answers with no parsable 'Answer: X': {bad}")


if __name__ == "__main__":
    main()