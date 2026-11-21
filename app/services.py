"""Shared logic for the REST API (app/main.py) and the Gradio demo (app/ui.py).

Counting runs src/count_traffic.py in a subprocess, so the evaluated counting logic is used unchanged.
"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = os.environ.get("WEIGHTS", str(ROOT / "models" / "final.pt"))
COUNTER = os.environ.get("COUNTER_SCRIPT", str(ROOT / "src" / "count_traffic.py"))
DEVICE = os.environ.get("DEVICE", "cpu")                       # "0" = first CUDA GPU
IMGSZ = int(os.environ.get("IMGSZ", "1280"))                   # the model was trained at 1280
STRIDE = max(1, int(os.environ.get("STRIDE", "2")))            # process every Nth frame
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "60"))
MAX_IMAGE_MB = int(os.environ.get("MAX_IMAGE_MB", "10"))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "10"))       # only the first N seconds are analysed
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "900"))
JOB_TTL_S = int(os.environ.get("JOB_TTL_S", "1800"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", str(Path(tempfile.gettempdir()) / "traffic_jobs")))
JOB_ID_RE = re.compile(r"[0-9a-f]{12}")


class BadInput(Exception):
    pass


class TooLarge(BadInput):
    pass


class Busy(Exception):
    pass


job_lock = threading.Lock()          # one counting job at a time (the demo runs on a small CPU)
JOBS = {}                            # job_id -> {"status": running|done|error, "result": ..., "error": ..., "created": ...}
_model = None
_model_lock = threading.Lock()


def _dev():
    return int(DEVICE) if DEVICE.isdigit() else DEVICE


def limits():
    return {"max_upload_mb": MAX_UPLOAD_MB, "max_video_seconds": MAX_SECONDS, "frame_stride": STRIDE, "imgsz": IMGSZ}


def is_busy():
    return job_lock.locked()


def new_job_id():
    return uuid.uuid4().hex[:12]


def job_file(job_id, name):
    return JOBS_DIR / job_id / name


# ---------------------------------------------------------------- single-image detection
def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(WEIGHTS)
    return _model


def detect_image(data: bytes, conf: float = 0.25):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise BadInput("could not decode the image (use JPEG or PNG)")
    model = get_model()
    r = model.predict(img, conf=conf, imgsz=IMGSZ, device=_dev(), verbose=False, agnostic_nms=True)[0]
    dets = []
    for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        dets.append({"class": model.names[c], "confidence": round(float(s), 3),
                     "box_xyxy": [round(float(v), 1) for v in box]})
    h, w = img.shape[:2]
    return {"image_size": [w, h], "count": len(dets), "detections": dets}


# ---------------------------------------------------------------- video helpers
def save_upload(upload, dest: Path, max_mb: int = MAX_UPLOAD_MB):
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit, total = max_mb * 1024 * 1024, 0
    with open(dest, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                break
            f.write(chunk)
    if total > limit:
        dest.unlink(missing_ok=True)
        raise TooLarge(f"file is larger than {max_mb} MB")
    if total == 0:
        dest.unlink(missing_ok=True)
        raise BadInput("empty file")
    return dest


def _run(cmd, timeout, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timed out after {timeout} s")
    if p.returncode != 0:
        raise RuntimeError(((p.stderr or p.stdout or "").strip()[-500:]) or f"exit code {p.returncode}")
    return p


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise BadInput("could not open the video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0 or frames <= 0:
        raise BadInput("the file does not contain a readable video")
    return {"fps": fps, "frames": frames, "width": w, "height": h, "seconds": frames / fps}


def prepare_video(src, dst):
    """Keep the first MAX_SECONDS, cap the width at 1920 (the scale the counter was evaluated at), re-encode as H.264."""
    try:
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", str(MAX_SECONDS), "-vf", "scale='min(1920,iw)':-2", "-an",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)], 300)
    except RuntimeError as e:
        print("prepare_video failed:", e, file=sys.stderr, flush=True)
        raise BadInput("could not read the file as a video (use mp4, mov, avi, mkv or webm)")


def make_geometry(line_y=0.5, roi_top=0.25, roi_bottom=1.0, x0=0.03, x1=0.97):
    """Horizontal counting line drawn left to right: the '+' side is below the line (toward the camera)."""
    if not (0.0 <= roi_top < line_y < roi_bottom <= 1.0):
        raise BadInput("the counting zone must contain the line: need roi_top < line_y < roi_bottom (all between 0 and 1)")
    return {"line": [[x0, line_y], [x1, line_y]],
            "roi": [[0.0, roi_top], [1.0, roi_top], [1.0, roi_bottom], [0.0, roi_bottom]]}


def validate_geometry(g):
    try:
        line, roi = g["line"], g["roi"]
        ok = (len(line) == 2 and len(roi) >= 3 and len(roi) <= 50
              and all(len(p) == 2 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p) for p in line + roi))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise BadInput('geometry must look like {"line": [[x,y],[x,y]], "roi": [[x,y], ...]} with values in 0..1')
    return {"line": [list(map(float, p)) for p in line], "roi": [list(map(float, p)) for p in roi]}


def preview_frame(video_path, line_y, roi_top, roi_bottom=1.0):
    cap = cv2.VideoCapture(str(video_path))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise BadInput("could not read the first frame")
    roi_top = min(roi_top, line_y - 0.02)
    g = make_geometry(line_y, max(roi_top, 0.0), roi_bottom)
    h, w = fr.shape[:2]
    W = 960
    fr = cv2.resize(fr, (W, int(h * W / w)))
    H = fr.shape[0]
    pts = np.array([[x * W, y * H] for x, y in g["roi"]], dtype=np.int32)
    cv2.polylines(fr, [pts], True, (0, 200, 255), 2)
    (x0, y0), (x1, y1) = [(int(x * W), int(y * H)) for x, y in g["line"]]
    cv2.line(fr, (x0, y0), (x1, y1), (0, 0, 255), 3)
    for txt, dy in (("below = toward camera (+)", 28), ("above = away from camera (-)", -12)):
        cv2.putText(fr, txt, (12, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(fr, txt, (12, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- counting jobs
def summarize_events(events_csv):
    by_class = defaultdict(lambda: {"to_plus": 0, "to_minus": 0})
    dups = 0
    with open(events_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "counted":
                by_class[r["cls"]][r["direction"]] += 1
            else:
                dups += 1
    plus = sum(v["to_plus"] for v in by_class.values())
    minus = sum(v["to_minus"] for v in by_class.values())
    return {"by_class": {k: dict(v) for k, v in sorted(by_class.items())},
            "totals": {"to_plus": plus, "to_minus": minus, "all": plus + minus},
            "duplicates_suppressed": dups}


def run_count(job_id, upload_path, geometry, only_dir="both", conf=0.25):
    t0 = time.perf_counter()
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    prepared = job_dir / "input.mp4"
    prepare_video(upload_path, prepared)
    info = probe(prepared)
    (job_dir / "geometry.json").write_text(json.dumps(geometry))
    analysed_s = min(info["seconds"], MAX_SECONDS)
    max_frames = max(1, int(np.ceil(analysed_s * info["fps"] / STRIDE)))

    cmd = [sys.executable, COUNTER, "--video", str(prepared), "--weights", WEIGHTS,
           "--geometry", str(job_dir / "geometry.json"), "--device", DEVICE, "--imgsz", str(IMGSZ),
           "--stride", str(STRIDE), "--conf", str(conf), "--max_frames", str(max_frames),
           "--out_width", "960", "--outdir", str(job_dir), "--no_review"]
    if only_dir in ("to_plus", "to_minus"):
        cmd += ["--only_dir", only_dir]
    p = _run(cmd, JOB_TIMEOUT_S, cwd=str(ROOT))

    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(job_dir / "count_input.mp4"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-movflags", "+faststart", str(job_dir / "annotated.mp4")], 300)
    (job_dir / "count_input.mp4").unlink(missing_ok=True)
    (job_dir / "events.csv").write_bytes((job_dir / "count_input_events.csv").read_bytes())
    m = re.search(r"(\d+) frames in ([\d.]+) s", p.stdout)
    return {
        "job_id": job_id,
        "video_seconds_analysed": round(analysed_s, 1),
        "frames_processed": int(m.group(1)) if m else None,
        "settings": {"frame_stride": STRIDE, "imgsz": IMGSZ, "device": DEVICE, "conf": conf, "only_dir": only_dir},
        "counts": summarize_events(job_dir / "events.csv"),
        "processing_seconds": round(time.perf_counter() - t0, 1),
        "files": {"video": f"/jobs/{job_id}/video", "events": f"/jobs/{job_id}/events"},
        "notes": ["to_plus = moving from above the line to below it (toward the camera) for the default horizontal line",
                  "counts are approximate: see the README for evaluation details and known failure cases"],
    }


def cleanup_jobs():
    now = time.time()
    for jid in [j for j, v in JOBS.items() if v["status"] != "running" and now - v["created"] > JOB_TTL_S]:
        JOBS.pop(jid, None)
    if JOBS_DIR.exists():
        for d in JOBS_DIR.iterdir():
            if d.is_dir() and now - d.stat().st_mtime > JOB_TTL_S and JOBS.get(d.name, {}).get("status") != "running":
                for f in d.iterdir():
                    f.unlink(missing_ok=True)
                d.rmdir()


def start_job(job_id, upload_path, geometry, only_dir="both", conf=0.25):
    """Run a counting job in a background thread. Raises Busy if another job is running."""
    cleanup_jobs()
    if not job_lock.acquire(blocking=False):
        raise Busy("another video is being processed, please try again in a minute")
    JOBS[job_id] = {"status": "running", "created": time.time()}

    def work():
        try:
            JOBS[job_id].update(status="done", result=run_count(job_id, upload_path, geometry, only_dir, conf))
        except BadInput as e:
            JOBS[job_id].update(status="error", error=str(e))
        except Exception:                            # details go to the server log, not to the client
            traceback.print_exc()
            JOBS[job_id].update(status="error", error="processing failed (see the server log)")
        finally:
            Path(upload_path).unlink(missing_ok=True)
            job_lock.release()

    threading.Thread(target=work, daemon=True).start()


def run_blocking(video_path, geometry, only_dir="both", conf=0.25):
    """Synchronous variant used by the Gradio demo."""
    cleanup_jobs()
    if not job_lock.acquire(timeout=3):
        raise Busy("another video is being processed, please try again in a minute")
    try:
        job_id = new_job_id()
        JOBS[job_id] = {"status": "running", "created": time.time()}
        res = run_count(job_id, Path(video_path), geometry, only_dir, conf)
        JOBS[job_id].update(status="done", result=res)
        return res
    finally:
        job_lock.release()
