import csv, json, subprocess, sys
from pathlib import Path

from eval_counts import load, match

CFG = Path("configs/count_eval.json")
VID = Path(r"D:\DATA\traffic_videos_1080")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    extra = sys.argv[2:]
    rows = []
    for item in json.loads(CFG.read_text()):
        clip, (lo, hi) = item["clip"], item["window"]
        cmd = [sys.executable, "src/count_traffic.py", "--video", str(VID / f"{clip}.mp4"),
               "--geometry", f"configs/geometry_{clip}.json", "--window", str(lo), str(hi),
               "--tag", tag] + extra + item.get("args", [])
        print(">>", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"!! {clip}: the counter failed, skipping this clip")
            continue
        man_path = Path(f"data/manual_{clip}.csv")
        if not man_path.exists():
            print(f"-- {clip}: no manual log at {man_path}, counted but not scored")
            continue
        det = [d for d in load(f"results/count_{clip}_{tag}_events.csv") if lo <= d["time_s"] <= hi]
        man = [m for m in load(str(man_path)) if lo <= m["time_s"] <= hi]
        pairs, fn, fp = match(man, det, 0.3)
        cls_ok = sum(m["cls"] == d["cls"] for m, d in pairs)
        rows.append({"clip": clip, "window": f"{lo}-{hi}s", "manual": len(man), "detected": len(det),
                     "matched": len(pairs), "missed": len(fn), "extra": len(fp),
                     "class_correct": f"{cls_ok}/{len(pairs)}"})

    if not rows:
        return
    out = Path("results") / f"counting_summary_{tag}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    cols = list(rows[0].keys())
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print("\nsaved", out)


if __name__ == "__main__":
    main()