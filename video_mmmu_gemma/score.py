"""
Live scoring of the Video-MMMU results written by native_sampling.py (safe to run while workers are running).
Re-scores every saved response with the official scorer in videommmu_official.py, so it always reflects the
current scoring code. Light enough for:   watch -n 30 uv run --no-sync python score.py
Add --save to also write OUT_DIR/_all_results.csv.
"""
import argparse, glob, json, os, time

import pandas as pd

import videommmu_official as vm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = f"{REPO_DIR}/dataset/videommmu"
OUT_DIR = f"{REPO_DIR}/results/videommmu_gemma4_12b_native"
TRACKS = ["Perception", "Comprehension", "Adaptation"]


def load_docs(data_dir):
    """{(id, track): doc} with only the columns scoring needs (skips the Adaptation images)."""
    docs = {}
    for track in TRACKS:
        for f in sorted(glob.glob(os.path.join(data_dir, track, "*.parquet"))):
            df = pd.read_parquet(f, columns=["id", "options", "answer", "question_type"])
            for d in df.to_dict("records"):
                d["options"] = [str(o) for o in (d["options"] if d["options"] is not None else [])]
                docs[(d["id"], track)] = d
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--save", action="store_true", help="write _all_results.csv")
    a = ap.parse_args()

    docs = load_docs(a.data_dir)
    total_videos = len({i for i, _ in docs})
    files = [f for f in glob.glob(os.path.join(a.out_dir, "*.csv")) if not os.path.basename(f).startswith("_")]
    print(time.strftime("%H:%M:%S"), f"  results: {a.out_dir}")
    if not files:
        print("No finished videos yet.")
        return

    allr = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    parsed, correct = [], []
    for r in allr.itertuples():
        p, c = vm.score(docs[(r.id, r.track)], r.response)
        parsed.append(p if isinstance(p, str) else json.dumps(p))
        correct.append(int(c))
    allr["parsed_pred"], allr["correct"] = parsed, correct

    # progress + ETA from the finish times of the most recent videos
    done = allr["id"].nunique()
    mtimes = sorted(os.path.getmtime(f) for f in files)
    recent = mtimes[-20:]
    line = f"Progress: {done}/{total_videos} videos ({100 * done / total_videos:.1f}%)   {len(allr)}/{3 * total_videos} questions"
    if len(recent) >= 2 and recent[-1] > recent[0] and done < total_videos:
        per_video = (recent[-1] - recent[0]) / (len(recent) - 1)
        eta = per_video * (total_videos - done)
        line += f"   ~{60 / per_video:.1f} videos/min   ETA ~{eta / 3600:.1f} h"
    line += f"   last result {int((time.time() - mtimes[-1]) / 60)} min ago"
    print(line)
    if done < total_videos:
        print("(partial run - accuracies are not final)")

    print(f"\nOverall accuracy: {100 * allr['correct'].mean():.2f}%  ({allr['correct'].sum()}/{len(allr)})")
    by_track = allr.groupby("track")["correct"].agg(["mean", "sum", "count"]).reindex(TRACKS).dropna()
    for t, r in by_track.iterrows():
        print(f"  {t:<14} {100 * r['mean']:6.2f}%  ({int(r['sum'])}/{int(r['count'])})")
    print("\nBy domain:")
    for dom, r in allr.groupby("domain")["correct"].agg(["mean", "count"]).iterrows():
        print(f"  {dom:<30} {100 * r['mean']:6.2f}%  (n={int(r['count'])})")

    mc = allr["question_type"] == "multiple-choice"
    no_ans = (allr.loc[mc, "parsed_pred"] == "No Answer Found.").sum()
    print(f"\nMC answers with no parsable choice: {no_ans}/{mc.sum()}")
    print(f"Response length (chars): median {int(allr['response'].str.len().median())}, max {allr['response'].str.len().max()}")

    if a.save:
        allr.to_csv(os.path.join(a.out_dir, "_all_results.csv"), index=False)
        print(f"saved {os.path.join(a.out_dir, '_all_results.csv')}")


if __name__ == "__main__":
    main()
