import argparse, csv
from collections import Counter


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["time_s"] = float(r["time_s"])
    return sorted(rows, key=lambda r: r["time_s"])


def match(manual, det, tol):
    used, pairs, fn = set(), [], []
    for m in manual:
        best = None
        for j, d in enumerate(det):
            if j in used or d["direction"] != m["direction"]:
                continue
            dt = abs(d["time_s"] - m["time_s"])
            if dt > tol:
                continue
            key = (d["cls"] != m["cls"], dt)      # prefer the same class, then the closest time
            if best is None or key < best[0]:
                best = (key, j)
        if best:
            used.add(best[1])
            pairs.append((m, det[best[1]]))
        else:
            fn.append(m)
    fp = [d for j, d in enumerate(det) if j not in used]
    return pairs, fn, fp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)      # results/count_<clip>_events.csv
    p.add_argument("--manual", required=True)      # time_s,cls,direction
    p.add_argument("--window", type=float, nargs=2, default=None)
    p.add_argument("--tol", type=float, default=1.0)
    a = p.parse_args()

    det, manual = load(a.events), load(a.manual)
    if a.window:
        lo, hi = a.window
        det = [d for d in det if lo <= d["time_s"] <= hi]
        manual = [m for m in manual if lo <= m["time_s"] <= hi]

    pairs, fn, fp = match(manual, det, a.tol)
    tp = len(pairs)
    cls_ok = sum(m["cls"] == d["cls"] for m, d in pairs)
    print(f"manual crossings: {len(manual)}   detected crossings: {len(det)}")
    print(f"matched: {tp}   missed: {len(fn)}   extra: {len(fp)}")
    print(f"recall {tp / max(len(manual), 1):.2f}   precision {tp / max(len(det), 1):.2f}   "
          f"class correct among matched {cls_ok / max(tp, 1):.2f}")
    mc, dc = Counter(m["cls"] for m in manual), Counter(d["cls"] for d in det)
    print("\ncount by class (manual, detected):")
    for c in sorted(set(mc) | set(dc)):
        print(f"  {c:<14}{mc[c]:>4}{dc[c]:>5}   diff {dc[c] - mc[c]:+d}")
    print("\nmissed (time, class, direction):", [(m["time_s"], m["cls"], m["direction"]) for m in fn])
    print("extra (time, id, class, direction):", [(d["time_s"], d["id"], d["cls"], d["direction"]) for d in fp])
    print("wrong class (time, manual, detected):",
          [(m["time_s"], m["cls"], d["cls"]) for m, d in pairs if m["cls"] != d["cls"]])


if __name__ == "__main__":
    main()