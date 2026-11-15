import argparse, time
from collections import defaultdict
from pathlib import Path
import cv2
from ultralytics import YOLO

VEHICLES = [0, 1, 3, 4, 5]   # car, motorcycle, autorickshaw, bus, truck (rider=2 and person=6 left out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/final.pt")
    p.add_argument("--video", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=1)          # process every Nth frame
    p.add_argument("--out_width", type=int, default=1280)    # width of the saved video
    p.add_argument("--max_frames", type=int, default=0)      # 0 = whole video
    a = p.parse_args()

    model = YOLO(a.weights)
    names = model.names
    stem = Path(a.video).stem
    out_path = Path("results") / f"track_{stem}_conf{a.conf}.mp4"
    out_path.parent.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    writer = None
    ids_by_class = defaultdict(set)
    n_frames = 0
    t_start = time.perf_counter()

    stream = model.track(source=a.video, stream=True, tracker="bytetrack.yaml",
                         persist=True, imgsz=a.imgsz, conf=a.conf, classes=VEHICLES,
                         vid_stride=a.stride, device=0, verbose=False)
    for r in stream:
        frame = r.plot(line_width=3)
        if r.boxes.id is not None:
            for tid, c in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                ids_by_class[names[c]].add(tid)

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (a.out_width, int(h * a.out_width / w)))
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps / a.stride, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n_frames += 1
        if n_frames % 100 == 0:
            print(n_frames, "frames done", flush=True)
        if a.max_frames and n_frames >= a.max_frames:
            break

    elapsed = time.perf_counter() - t_start
    if writer is not None:
        writer.release()

    print(f"\n{n_frames} frames processed in {elapsed:.1f} s "
          f"({n_frames / elapsed:.1f} FPS including drawing and writing)")
    print(f"saved {out_path}")
    print("unique track IDs per class (an overcount, since IDs fragment):")
    for k, v in sorted(ids_by_class.items()):
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()