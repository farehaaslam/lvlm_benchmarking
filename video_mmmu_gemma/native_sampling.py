"""
Run Gemma 4 12B on Video-MMMU (all 900 questions) with the OFFICIAL lmms-eval prompts and scoring
(see videommmu_official.py), letting the model's own processor handle the video (no manual frame sampling).

Data (official HF release, lmms-lab/VideoMMMU), all inside this repo under DATA_DIR:
    DATA_DIR/Perception/test-*.parquet
    DATA_DIR/Comprehension/test-*.parquet
    DATA_DIR/Adaptation/test-*.parquet
    DATA_DIR/**/<id>.mp4          # Art.zip, Business.zip, ... extracted anywhere below DATA_DIR
  e.g.  hf download lmms-lab/VideoMMMU --repo-type dataset --local-dir dataset/videommmu
        cd dataset/videommmu && for z in Art Business Engineering Humanities Medicine Science; do unzip -q $z.zip; done

Adaptation: in the official videos the question image is the LAST frame, but the processor's uniform
sampler (index i*N/num_frames) never picks the last frame. So the dataset's `image` is passed right after
the video, which is the README's sanctioned alternative and keeps the official prompt
("The image for this question is at the end of the video") accurate.

Saves one CSV per video id (its Perception / Comprehension / Adaptation rows). A video is only saved when
all its questions ran without error, so a rerun retries failures and skips finished videos.
Run:  python native_sampling.py            (one process, model spread over all visible GPUs)
      4 workers x 2 GPUs (~4x faster, same results): for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$((2*i)),$((2*i+1)) nohup uv run --no-sync python native_sampling.py \\
          --shard_id $i --num_shards 4 > logs/worker_$i.log 2>&1 & done
      then: uv run python native_sampling.py --summary_only
      Test first with --limit_videos 3.
"""
import os

# ============================ CONFIG: EDIT THESE ============================
REPO_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = f"{REPO_DIR}/dataset/videommmu"              # parquet configs + extracted videos (see above)
VIDEO_DIR      = DATA_DIR                                      # searched recursively for <id>.mp4
OUT_DIR        = f"{REPO_DIR}/results/videommmu_gemma4_12b_native"
MODEL_ID       = "google/gemma-4-12B-it"
ENABLE_THINKING = True                        # official eval is non-thinking
USE_RECOMMENDED_SAMPLING = False              # False = greedy (repeatable). True = Google's temp 1.0 / top_p 0.95 / top_k 64
LIMIT_VIDEOS   = 0                            # 0 = all 300; set e.g. 3 for a test run
# ============================================================================

import argparse, gc, glob, io, json, re
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor

import videommmu_official as vm

try:
    from transformers import AutoModelForMultimodalLM as AutoModel   # class used on the Gemma 4 model card
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModel

TRACKS = ["Perception", "Comprehension", "Adaptation"]
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build_video_index(root):
    idx, dupes = {}, []
    for dp, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in VIDEO_EXT:
                if stem in idx:
                    dupes.append(stem)
                idx.setdefault(stem, os.path.join(dp, f))
    if dupes:
        print(f"WARNING: {len(dupes)} video names appear more than once, using the first: {dupes[:5]}")
    return idx


def load_questions(data_dir):
    """{video id: {track: doc}} from the official parquet files."""
    by_id = {}
    for track in TRACKS:
        files = sorted(glob.glob(os.path.join(data_dir, track, "*.parquet")))
        if not files:
            raise SystemExit(f"No parquet files in {os.path.join(data_dir, track)}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if df["id"].duplicated().any():
            raise SystemExit(f"{track}: duplicate ids {df.loc[df['id'].duplicated(), 'id'].tolist()[:5]}")
        for doc in df.to_dict("records"):
            doc["options"] = [str(o) for o in (doc.get("options") if doc.get("options") is not None else [])]
            by_id.setdefault(doc["id"], {})[track] = doc
    return by_id


def load_image(img):
    if img is None:
        return None
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path") and os.path.isfile(img["path"]):
            return Image.open(img["path"]).convert("RGB")
        return None
    return img.convert("RGB")


def clean_response(processor, raw, input_ids):
    """Strip Gemma's channel/thought tags the way the model card does; fall back to raw text."""
    try:
        parsed = processor.parse_response(raw, prefix=input_ids)
        return str(parsed.get("content") or "") if isinstance(parsed, dict) else str(parsed)
    except Exception:
        return raw


def rescore(allr, by_id):
    """Re-score saved responses with the current scorer (results don't need regenerating if scoring changes)."""
    parsed, correct = [], []
    for r in allr.itertuples():
        p, c = vm.score(by_id[r.id][r.track], "" if pd.isna(r.response) else r.response)
        parsed.append(p if isinstance(p, str) else json.dumps(p))
        correct.append(int(c))
    allr["parsed_pred"], allr["correct"] = parsed, correct
    return allr


def summarize(out_dir, by_id):
    expected_videos = len(by_id)
    parts = [pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(os.path.join(out_dir, "*.csv")))
             if not os.path.basename(f).startswith("_")]
    if not parts:
        print("No results yet.")
        return
    allr = rescore(pd.concat(parts, ignore_index=True), by_id)
    allr.to_csv(os.path.join(out_dir, "_all_results.csv"), index=False)
    n_vid = allr["id"].nunique()
    print(f"\nVideos: {n_vid}/{expected_videos}   Questions: {len(allr)}/{3 * expected_videos}")
    if n_vid < expected_videos:
        print("WARNING: run incomplete - numbers below are NOT comparable to the full benchmark.")
    print(f"Overall accuracy: {100 * allr['correct'].mean():.2f}%")
    print((100 * allr.groupby("track")["correct"].mean()).reindex(TRACKS).round(2).to_string())
    print((100 * allr.groupby("domain")["correct"].mean()).round(2).to_string())
    mc = allr["question_type"] == "multiple-choice"
    print(f"MC answers with no parsable choice: {(allr.loc[mc, 'parsed_pred'] == 'No Answer Found.').sum()} / {mc.sum()}")
    print(f"Median input tokens: {int(allr['input_tokens'].median())}   Max: {int(allr['input_tokens'].max())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--video_dir", default=VIDEO_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit_videos", type=int, default=LIMIT_VIDEOS)
    ap.add_argument("--shard_id", type=int, default=0, help="this worker's slice of the videos (with --num_shards)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--summary_only", action="store_true", help="just print the summary of OUT_DIR")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    by_id = load_questions(a.data_dir)
    incomplete = [i for i, t in by_id.items() if set(t) != set(TRACKS)]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} ids are missing a track, e.g. {incomplete[:5]}")
    ids = sorted(by_id, key=natural_key)
    expected = len(ids)

    vid_index = build_video_index(a.video_dir)
    missing = [i for i in ids if i not in vid_index]
    print(f"{expected} videos in the question files, {len(vid_index)} video files indexed, {len(missing)} missing")
    if missing:
        print(f"WARNING: no video file for {missing[:10]}{' ...' if len(missing) > 10 else ''} - these are skipped")
    if a.summary_only:
        summarize(a.out_dir, by_id)
        return
    if a.limit_videos:
        ids = ids[: a.limit_videos]
    ids = ids[a.shard_id :: a.num_shards]
    print(f"shard {a.shard_id}/{a.num_shards}: {len(ids)} videos")

    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype="auto", device_map="auto").eval()

    gen_kwargs = dict(max_new_tokens=vm.MAX_NEW_TOKENS)
    if USE_RECOMMENDED_SAMPLING:
        gen_kwargs.update(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
    else:
        gen_kwargs.update(do_sample=False)

    failed = []
    for n, vid in enumerate(ids, 1):
        out_path = os.path.join(a.out_dir, f"{vid}.csv")
        if os.path.exists(out_path) or vid not in vid_index:
            continue
        vpath = vid_index[vid]

        rows, error = [], None
        for track in TRACKS:
            doc = by_id[vid][track]
            try:
                # model card: video (and images) go BEFORE the text; the processor handles the video itself
                content = [{"type": "video", "video": vpath}]
                image = load_image(doc.get("image")) if track == "Adaptation" else None
                if image is not None:
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": vm.doc_to_text(doc, track)})

                inputs = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True, enable_thinking=ENABLE_THINKING,
                ).to(model.device)
                in_len = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
                raw = processor.decode(out[0][in_len:], skip_special_tokens=False)
                resp = vm.strip_special_tokens(clean_response(processor, raw, inputs["input_ids"]))
                del inputs, out
            except Exception as e:
                error = f"{track}: {type(e).__name__}: {e}"
                break
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            parsed, correct = vm.score(doc, resp)
            subject = vm.extract_subset_name(vid)
            rows.append({
                "id": vid, "track": track, "subject": subject, "domain": vm.SUB_CAT2DOMAIN.get(subject, ""),
                "question_type": doc["question_type"], "qa_type": doc.get("qa_type", ""),
                "answer": doc["answer"], "parsed_pred": parsed if isinstance(parsed, str) else json.dumps(parsed),
                "correct": int(correct), "input_tokens": in_len, "used_question_image": int(image is not None),
                "response": resp,
            })

        if error:
            failed.append(vid)
            print(f"[{n}/{len(ids)}] {vid}: NOT SAVED ({error}) - will retry on next run", flush=True)
            continue
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[{n}/{len(ids)}] {vid}: " + ", ".join(f"{r['track'][:4]}={r['correct']}" for r in rows) +
              f"  tokens={rows[0]['input_tokens']}", flush=True)

    if failed:
        print(f"\n{len(failed)} videos failed this run and were not saved: {failed}")
    if a.num_shards == 1:  # with shards, run --summary_only once every worker is done
        summarize(a.out_dir, by_id)


if __name__ == "__main__":
    main()
