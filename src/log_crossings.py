import argparse, csv, json, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

KEYS = {"c": "car", "m": "motorcycle", "a": "autorickshaw", "b": "bus", "t": "truck"}
OTHER = {"to_plus": "to_minus", "to_minus": "to_plus"}
HELP = "c car  m motorcycle  a auto  b bus  t truck | SHIFT = other direction | d flip default | u undo | space pause | , . step | j l jump 2s | [ ] speed | q save+quit"


def load_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{"time_s": float(r["time_s"]), "cls": r["cls"], "direction": r["direction"]}
                for r in csv.DictReader(f)]


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["time_s", "cls", "direction"])
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["time_s"]))


def put(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--geometry", required=True)           # same JSON the counter uses
    p.add_argument("--out", default=None)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)      # 0 = until the end of the clip
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--default_dir", choices=["to_plus", "to_minus"], default="to_minus")
    p.add_argument("--lag", type=float, default=0.3)      # your reaction time in real seconds
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--fresh", action="store_true")        # ignore an existing log file
    a = p.parse_args()

    out = Path(a.out) if a.out else Path("data") / f"manual_{Path(a.video).stem}.csv"
    rows = [] if a.fresh else load_rows(out)
    geom = json.loads(Path(a.geometry).read_text())

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0, H0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    DW = a.width
    DH = int(H0 * DW / W0)
    A = (geom["line"][0][0] * DW, geom["line"][0][1] * DH)
    B = (geom["line"][1][0] * DW, geom["line"][1][1] * DH)
    roi = np.array([[x * DW, y * DH] for x, y in geom["roi"]], dtype=np.int32)
    nx, ny = -(B[1] - A[1]), (B[0] - A[0])
    nl = max(np.hypot(nx, ny), 1e-6)
    nx, ny = nx / nl, ny / nl                              # same "+" side as the counter
    mid = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)

    def seek(k):
        k = int(min(max(k, 0), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        return (k, fr) if ok else (None, None)

    pos, frame = seek(int(a.start * fps))
    assert frame is not None, "could not read the start frame"
    end_frame = int(a.end * fps) if a.end else total - 1
    paused, default_dir, speed, msg, n_new = False, a.default_dir, a.speed, "", 0

    cv2.namedWindow("log crossings", cv2.WINDOW_AUTOSIZE)
    while True:
        t_loop = time.perf_counter()
        view = cv2.resize(frame, (DW, DH))
        cv2.polylines(view, [roi], True, (0, 200, 255), 1)
        cv2.line(view, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (0, 0, 255), 2)
        for s, t in ((1, "+"), (-1, "-")):
            put(view, t, (int(mid[0] + s * nx * 40) - 8, int(mid[1] + s * ny * 40) + 8), 1.0, (0, 0, 255))
        put(view, f"t={pos / fps:6.2f}s  {speed:.2f}x  {'PAUSED' if paused else 'playing'}   "
                  f"logged this session: {n_new}   in file: {len(rows)}", (10, 24))
        put(view, f"default direction: {default_dir}   (Shift+key = {OTHER[default_dir]})", (10, 48))
        if msg:
            put(view, msg, (10, 72), 0.6, (0, 255, 255))
        put(view, HELP, (10, DH - 12), 0.45)
        cv2.imshow("log crossings", view)

        if paused:
            key = cv2.waitKey(50) & 0xFF
        else:
            wait = max(1, int(1000 / (fps * speed) - (time.perf_counter() - t_loop) * 1000))
            key = cv2.waitKey(wait) & 0xFF

        ch = chr(key) if 32 <= key < 127 else ""
        if key == 27 or ch == "q":
            break
        elif ch == " ":
            paused = not paused
        elif ch == "d":
            default_dir = OTHER[default_dir]
        elif ch == "u" and n_new > 0:
            gone = rows.pop()
            n_new -= 1
            save_rows(out, rows)
            msg = f"undid: {gone['time_s']}s {gone['cls']} {gone['direction']}"
        elif ch == "[":
            speed = max(0.1, speed / 1.5)
        elif ch == "]":
            speed = min(2.0, speed * 1.5)
        elif ch == "," and paused:
            pos, fr = seek(pos - 1)
            frame = fr if fr is not None else frame
        elif ch == "." and paused:
            ok, fr = cap.read()
            if ok:
                pos, frame = pos + 1, fr
        elif ch == "j":
            pos, fr = seek(pos - int(2 * fps))
            frame = fr if fr is not None else frame
        elif ch == "l":
            pos, fr = seek(pos + int(2 * fps))
            frame = fr if fr is not None else frame
        elif ch.lower() in KEYS:
            d = OTHER[default_dir] if ch.isupper() else default_dir
            t_log = max(0.0, pos / fps - (0.0 if paused else a.lag * speed))
            row = {"time_s": round(t_log, 2), "cls": KEYS[ch.lower()], "direction": d}
            rows.append(row)
            n_new += 1
            save_rows(out, rows)
            msg = f"logged: {row['time_s']}s {row['cls']} {row['direction']}"

        if not paused:
            if pos >= end_frame:
                break
            ok, fr = cap.read()
            if not ok:
                break
            pos, frame = pos + 1, fr

    cv2.destroyAllWindows()
    cap.release()
    save_rows(out, rows)
    cnt = Counter((r["cls"], r["direction"]) for r in rows)
    print(f"saved {len(rows)} crossings to {out}")
    for (c, d), v in sorted(cnt.items()):
        print(f"  {c:<14}{d:<10}{v}")


if __name__ == "__main__":
    main()