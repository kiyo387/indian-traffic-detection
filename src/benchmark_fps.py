import argparse, time
import cv2
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--half", action="store_true")      # fp16 on GPU
    a = p.parse_args()

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    ok, frame = cap.read()
    assert ok, "could not read the video"
    print("video frame size:", frame.shape[1], "x", frame.shape[0])

    for _ in range(10):                                  # warm-up, not timed
        model.predict(frame, imgsz=a.imgsz, device=0, half=a.half, verbose=False)

    t_model = t_total = 0.0
    n = 0
    while n < a.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        model.predict(frame, imgsz=a.imgsz, device=0, half=a.half, verbose=False)
        t2 = time.perf_counter()
        t_model += t2 - t1
        t_total += t2 - t0
        n += 1

    print(f"{n} frames, imgsz={a.imgsz}, half={a.half}")
    print(f"model only : {n / t_model:.1f} FPS ({1000 * t_model / n:.1f} ms/frame)")
    print(f"end to end : {n / t_total:.1f} FPS (includes video decode)")


if __name__ == "__main__":
    main()