import argparse, csv, json, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLES = [0, 1, 3, 4, 5]          # car, motorcycle, autorickshaw, bus, truck
DEFAULT_GEOMETRY = {"line": [[0.05, 0.50], [0.95, 0.50]],
                    "roi": [[0.0, 0.25], [1.0, 0.25], [1.0, 1.0], [0.0, 1.0]]}


def side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing(prev, cur, a, b):
    """+1: moved from the minus side to the plus side of line a-b; -1: the reverse; 0: no crossing."""
    s1, s2 = side(prev, a, b), side(cur, a, b)
    if (s1 < 0) == (s2 < 0):
        return 0
    k = s1 / (s1 - s2)
    hit = (prev[0] + (cur[0] - prev[0]) * k, prev[1] + (cur[1] - prev[1]) * k)
    ab = (b[0] - a[0], b[1] - a[1])
    t = ((hit[0] - a[0]) * ab[0] + (hit[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)
    if not 0.0 <= t <= 1.0:
        return 0                     # crossed the extended line, not the drawn segment
    return 1 if s2 >= 0 else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="models/final.pt")
    p.add_argument("--geometry", default=None)           # JSON from pick_geometry.py
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--min_age", type=int, default=5)      # frames a track must be seen in the ROI before it can count
    p.add_argument("--max_jump", type=float, default=150) # px; ignore bigger jumps between frames
    p.add_argument("--window", type=float, nargs=2, default=None)   # e.g. --window 10 30 (seconds)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--out_width", type=int, default=1280)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    geom = json.loads(Path(a.geometry).read_text()) if a.geometry else DEFAULT_GEOMETRY
    model = YOLO(a.weights)
    names = model.names
    stem = Path(a.video).stem + (f"_{a.tag}" if a.tag else "")
    Path("results").mkdir(exist_ok=True)
    out_video = Path("results") / f"count_{stem}.mp4"
    out_csv = Path("results") / f"count_{stem}_events.csv"

    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    stream = model.track(source=a.video, stream=True, tracker=a.tracker, persist=True, imgsz=a.imgsz,
                         conf=a.conf, classes=VEHICLES, vid_stride=a.stride, device=0, verbose=False)

    A = B = roi = writer = None
    last_pt, age, votes, counted = {}, defaultdict(int), defaultdict(Counter), {}
    counts, events, crops = Counter(), [], []
    n, t0 = 0, time.perf_counter()

    for r in stream:
        frame = r.orig_img.copy()
        h, w = frame.shape[:2]
        if A is None:
            A = (geom["line"][0][0] * w, geom["line"][0][1] * h)
            B = (geom["line"][1][0] * w, geom["line"][1][1] * h)
            roi = np.array([[x * w, y * h] for x, y in geom["roi"]], dtype=np.float32)
            fs = w / 1920.0
        t_s = n * a.stride / src_fps

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            cls = r.boxes.cls.int().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, c, bb in zip(ids, cls, xyxy):
                pt = (float((bb[0] + bb[2]) / 2), float(bb[3]))
                if cv2.pointPolygonTest(roi.reshape(-1, 1, 2), pt, False) < 0:
                    last_pt.pop(tid, None)
                    continue
                age[tid] += 1
                votes[tid][c] += 1
                prev, last_pt[tid] = last_pt.get(tid), pt
                done = tid in counted
                color = (0, 255, 255) if done else (0, 200, 0)
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, max(1, int(3 * fs)))
                cv2.circle(frame, (int(pt[0]), int(pt[1])), max(2, int(5 * fs)), (0, 0, 255), -1)
                cv2.putText(frame, f"{tid} {names[c]}", (int(bb[0]), max(15, int(bb[1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * fs + 0.2, color, max(1, int(2 * fs)))

                if (prev is not None and not done and age[tid] >= a.min_age
                        and np.hypot(pt[0] - prev[0], pt[1] - prev[1]) <= a.max_jump):
                    d = crossing(prev, pt, A, B)
                    if d:
                        cname = names[votes[tid].most_common(1)[0][0]]
                        counted[tid] = d
                        counts[(cname, d)] += 1
                        events.append({"frame": n * a.stride, "time_s": round(t_s, 2), "id": tid,
                                       "cls": cname, "direction": "to_plus" if d > 0 else "to_minus"})
                        x1, y1, x2, y2 = [int(max(0, v)) for v in bb]
                        crop = r.orig_img[y1:y2, x1:x2]
                        if crop.size:
                            crop = cv2.resize(crop, (max(40, int(crop.shape[1] * 150 / crop.shape[0])), 150))
                            crops.append((crop, f"{t_s:.1f}s #{tid} {cname} {'+' if d > 0 else '-'}"))
        # overlay: ROI, line, +/- marks, counts panel
        cv2.polylines(frame, [roi.astype(np.int32)], True, (0, 200, 255), max(1, int(2 * fs)))
        cv2.line(frame, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (0, 0, 255), max(2, int(4 * fs)))
        mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        nx, ny = -(B[1] - A[1]), (B[0] - A[0])
        nl = max(np.hypot(nx, ny), 1e-6)
        for sign, txt in ((1, "+"), (-1, "-")):
            px, py = mx + sign * nx / nl * 60 * fs, my + sign * ny / nl * 60 * fs
            cv2.putText(frame, txt, (int(px) - 10, int(py) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.6 * fs, (0, 0, 255), max(2, int(4 * fs)))
        classes = sorted({c for c, _ in counts})
        rows = ["COUNTS  to +  /  to -"] + [f"{c}: {counts[(c, 1)]} / {counts[(c, -1)]}" for c in classes]
        rows.append(f"total: {sum(v for (c, d), v in counts.items() if d > 0)} / {sum(v for (c, d), v in counts.items() if d < 0)}")
        ph = int((len(rows) * 34 + 14) * fs)
        sub = frame[0:ph, 0:int(430 * fs)]
        frame[0:ph, 0:int(430 * fs)] = (sub * 0.35).astype(np.uint8)
        for i, t in enumerate(rows):
            cv2.putText(frame, t, (int(12 * fs), int((34 * (i + 1)) * fs)), cv2.FONT_HERSHEY_SIMPLEX, 0.95 * fs, (255, 255, 255), max(1, int(2 * fs)))

        out = cv2.resize(frame, (a.out_width, int(h * a.out_width / w)))
        if writer is None:
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / a.stride,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
        n += 1
        if n % 200 == 0:
            print(n, "frames,", len(events), "crossings so far", flush=True)
        if a.max_frames and n >= a.max_frames:
            break

    elapsed = time.perf_counter() - t0
    if writer is not None:
        writer.release()
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "id", "cls", "direction"])
        wr.writeheader()
        wr.writerows(events)
        if crops:
            cols, cw, ch = 8, 200, 172
            nrows = (len(crops) + cols - 1) // cols
            sheet = np.full((nrows * ch, cols * cw, 3), 30, np.uint8)
            for i, (cr, lab) in enumerate(crops):
                ry, cx = divmod(i, cols)
                w_ = min(cr.shape[1], cw)
                sheet[ry * ch + 22: ry * ch + 172, cx * cw: cx * cw + w_] = cr[:, :w_]
                cv2.putText(sheet, lab, (cx * cw + 2, ry * ch + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(Path("results") / f"count_{stem}_review.jpg"), sheet)

    def table(title, cnt):
        print(f"\n{title}\n{'class':<14}{'to +':>6}{'to -':>6}{'total':>7}")
        for c in sorted({c for c, _ in cnt}):
            print(f"{c:<14}{cnt[(c, 1)]:>6}{cnt[(c, -1)]:>6}{cnt[(c, 1)] + cnt[(c, -1)]:>7}")
        tp = sum(v for (c, d), v in cnt.items() if d > 0)
        tm = sum(v for (c, d), v in cnt.items() if d < 0)
        print(f"{'ALL':<14}{tp:>6}{tm:>6}{tp + tm:>7}")

    table("Whole video", counts)
    if a.window:
        lo, hi = a.window
        wc = Counter((e["cls"], 1 if e["direction"] == "to_plus" else -1)
                     for e in events if lo <= e["time_s"] <= hi)
        table(f"Window {lo}s to {hi}s", wc)
    print(f"\n{n} frames in {elapsed:.1f} s ({n / elapsed:.1f} FPS including drawing and writing)")
    print(f"track IDs seen inside the ROI: {len(age)}, of which counted: {len(counted)}")
    print("saved", out_video, "and", out_csv)


if __name__ == "__main__":
    main()