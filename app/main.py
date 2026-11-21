"""FastAPI service: /detect (image), /count (video, asynchronous job), /jobs/..., plus the Gradio demo at /demo."""
import json
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from . import services as S
from .ui import build_demo

app = FastAPI(
    title="Indian Traffic Detection and Counting",
    version="1.0.0",
    description="YOLOv8 detector (trained on IDD) with ByteTrack tracking and line-crossing vehicle counts. "
                "Demo limits apply: short clips, one video at a time.",
)


def _bad(e: S.BadInput):
    return HTTPException(413 if isinstance(e, S.TooLarge) else 400, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "device": S.DEVICE, "weights": Path(S.WEIGHTS).name,
            "weights_present": Path(S.WEIGHTS).exists(), "busy": S.is_busy(), "limits": S.limits()}


@app.post("/detect")
def detect(file: UploadFile = File(...), conf: float = Form(0.25)):
    """Detect vehicles in one image (JPEG or PNG). Returns boxes in pixels."""
    if not 0.01 <= conf <= 0.99:
        raise HTTPException(400, "conf must be between 0.01 and 0.99")
    limit = S.MAX_IMAGE_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"image is larger than {S.MAX_IMAGE_MB} MB")
    try:
        return S.detect_image(data, conf)
    except S.BadInput as e:
        raise _bad(e)


@app.post("/count", status_code=202)
def count(file: UploadFile = File(...), line_y: float = Form(0.5), roi_top: float = Form(0.25),
          roi_bottom: float = Form(1.0), only_dir: str = Form("both"), conf: float = Form(0.25),
          geometry: str = Form("")):
    """Start a counting job on a video. Poll GET /jobs/{job_id} until status is 'done'.

    By default a horizontal line is used: 'to_plus' means moving down the image (toward the camera).
    Pass `geometry` as JSON ({"line": [[x,y],[x,y]], "roi": [[x,y], ...]}, normalised 0..1) for a custom line.
    """
    if only_dir not in ("both", "to_plus", "to_minus"):
        raise HTTPException(400, "only_dir must be both, to_plus or to_minus")
    if not 0.01 <= conf <= 0.99:
        raise HTTPException(400, "conf must be between 0.01 and 0.99")
    if S.is_busy():
        raise HTTPException(429, "another video is being processed, please try again in a minute",
                            headers={"Retry-After": "30"})
    try:
        geom = S.validate_geometry(json.loads(geometry)) if geometry.strip() else S.make_geometry(line_y, roi_top, roi_bottom)
    except json.JSONDecodeError:
        raise HTTPException(400, "geometry is not valid JSON")
    except S.BadInput as e:
        raise _bad(e)

    job_id = S.new_job_id()
    suffix = Path(file.filename or "").suffix.lower()
    suffix = suffix if suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm") else ".mp4"
    try:
        path = S.save_upload(file, S.job_file(job_id, "upload" + suffix))
        S.start_job(job_id, path, geom, only_dir, conf)
    except S.BadInput as e:
        raise _bad(e)
    except S.Busy as e:
        raise HTTPException(429, str(e), headers={"Retry-After": "30"})
    return {"job_id": job_id, "status": "running", "status_url": f"/jobs/{job_id}"}


def _job(job_id):
    if not S.JOB_ID_RE.fullmatch(job_id) or job_id not in S.JOBS:
        raise HTTPException(404, "unknown or expired job")
    return S.JOBS[job_id]


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = _job(job_id)
    return {"job_id": job_id, **{k: v for k, v in j.items() if k != "created"}}


def _file(job_id, name, media):
    j = _job(job_id)
    path = S.job_file(job_id, name)
    if j["status"] != "done" or not path.exists():
        raise HTTPException(409, f"job is {j['status']}; the file is not available")
    return FileResponse(path, media_type=media, filename=name)


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    return _file(job_id, "annotated.mp4", "video/mp4")


@app.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    return _file(job_id, "events.csv", "text/csv")


app = gr.mount_gradio_app(app, build_demo(), path="/")
