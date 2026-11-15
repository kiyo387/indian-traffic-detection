import csv, time
from pathlib import Path
import cv2, torch
from ultralytics import YOLO

VID = Path(r"D:\DATA\traffic_videos_1080")
CLIPS = ["test4", "test24", "test6"]
MODELS = {
    "yolov8n_1280": ("runs/detect/idd-traffic/imgsz1280/weights/best.pt", 1280),
    "yolov8s_640": ("runs/detect/idd-traffic/yolov8s/weights/best.pt", 640),
}
FRAMES = 200      # frames timed per run (test24 has 299)
WARMUP = 10       # untimed frames, so GPU start-up cost is excluded


def fp16_kwargs(api):
    """Arguments that request FP16 inference, for either API name."""
    return {"quantize": 16} if api == "quantize" else {"half": True}


def bench(weights, imgsz, half, clip_path):
    model = YOLO(weights)                      # fresh model object for every run
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {clip_path}")
    h, w = frame.shape[:2]

    # decide how to ask for FP16 (new quantize argument first, old half as fallback)
    api = "fp32"
    prec = {}
    if half:
        api = "quantize"
        prec = fp16_kwargs(api)
        try:
            model.predict(frame, imgsz=imgsz, device=0, verbose=False, **prec)
        except Exception as e:
            print(f"  quantize not accepted here ({type(e).__name__}), using half=True instead")
            api = "half"
            prec = fp16_kwargs(api)

    for _ in range(WARMUP):
        model.predict(frame, imgsz=imgsz, device=0, verbose=False, **prec)
    torch.cuda.synchronize()

    t_dec = t_pred = 0.0
    pre = inf = post = 0.0
    dets = n = 0
    while n < FRAMES:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        r = model.predict(frame, imgsz=imgsz, device=0, verbose=False, **prec)[0]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_dec += t1 - t0
        t_pred += t2 - t1
        pre += r.speed["preprocess"]
        inf += r.speed["inference"]
        post += r.speed["postprocess"]
        dets += len(r.boxes)
        n += 1
    cap.release()

    return {
        "clip": clip_path.stem, "resolution": f"{w}x{h}", "imgsz": imgsz,
        "precision": "fp16" if half else "fp32", "fp16_api": api,
        "frames": n,
        "decode_ms": round(1000 * t_dec / n, 1),
        "predict_ms": round(1000 * t_pred / n, 1),
        "pre_ms": round(pre / n, 1),
        "infer_ms": round(inf / n, 1),
        "post_ms": round(post / n, 1),
        "fps_model": round(n / t_pred, 1),
        "fps_end2end": round(n / (t_dec + t_pred), 1),
        "dets_per_frame": round(dets / n, 1),
    }


def main():
    rows = []
    Path("results").mkdir(exist_ok=True)
    for clip in CLIPS:
        matches = list(VID.glob(clip + ".*"))
        if not matches:
            print(f"skipping {clip}: not found in {VID}")
            continue
        path = matches[0]
        for name, (weights, imgsz) in MODELS.items():
            for half in (False,):
                row = {"model": name, **bench(weights, imgsz, half, path)}
                rows.append(row)
                print(row["clip"], row["model"], row["precision"],
                      "| decode", row["decode_ms"], "ms",
                      "| predict", row["predict_ms"], "ms",
                      "| end-to-end", row["fps_end2end"], "FPS", flush=True)
                with open("results/fps_benchmark_1080.csv", "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    wr.writeheader()
                    wr.writerows(rows)
    print("saved results/fps_benchmark_1080.csv")


if __name__ == "__main__":
    main()