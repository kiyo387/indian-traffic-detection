"""Gradio demo: upload a short clip, place the counting line, get counts and an annotated video."""
import traceback

import gradio as gr

from . import services as S

DIRECTIONS = [("Both directions", "both"),
              ("Only toward the camera (down the image)", "to_plus"),
              ("Only away from the camera (up the image)", "to_minus")]

INTRO = f"""
# Indian Traffic Detection and Counting
A YOLOv8 detector trained on the Indian Driving Dataset (IDD), with ByteTrack tracking and line-crossing counts.

**Demo limits:** the first {S.MAX_SECONDS:g} seconds of a clip (up to {S.MAX_UPLOAD_MB} MB), every {S.STRIDE} frames, on a small CPU,
one video at a time. Expect a minute or two. Fixed cameras looking along a road (bridge or footpath views) work best.
Counts are approximate; per-class counts for autorickshaws and trucks are the least reliable.
"""


def preview_fn(video, line_y, roi_top):
    if not video:
        return None
    try:
        return S.preview_frame(video, line_y, roi_top)
    except S.BadInput as e:
        raise gr.Error(str(e))


def count_fn(video, line_y, roi_top, direction, conf):
    if not video:
        raise gr.Error("Upload a video first.")
    try:
        res = S.run_blocking(video, S.make_geometry(line_y, roi_top, 1.0), direction, conf)
    except (S.BadInput, S.Busy) as e:
        raise gr.Error(str(e))
    except Exception:
        traceback.print_exc()
        raise gr.Error("Processing failed (see the server log).")
    by_class = res["counts"]["by_class"]
    rows = [[c, v["to_plus"], v["to_minus"], v["to_plus"] + v["to_minus"]] for c, v in by_class.items()]
    t = res["counts"]["totals"]
    rows.append(["ALL", t["to_plus"], t["to_minus"], t["all"]])
    summary = (f"Analysed {res['video_seconds_analysed']} s of video ({res['frames_processed']} frames) "
               f"in {res['processing_seconds']} s. Suppressed {res['counts']['duplicates_suppressed']} duplicate tracks.")
    return str(S.job_file(res["job_id"], "annotated.mp4")), rows, summary


def build_demo():
    with gr.Blocks(title="Indian Traffic Detection and Counting") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Traffic video", sources=["upload"])
                line_y = gr.Slider(0.10, 0.95, value=0.50, step=0.01,
                                   label="Counting line height (0 = top of the image, 1 = bottom)")
                roi_top = gr.Slider(0.0, 0.90, value=0.25, step=0.01,
                                    label="Top of the counting zone (ignores the far distance above it)")
                direction = gr.Radio(DIRECTIONS, value="both", label="Directions to count")
                conf = gr.Slider(0.10, 0.60, value=0.25, step=0.05, label="Detection confidence threshold")
                button = gr.Button("Count vehicles", variant="primary")
            preview = gr.Image(label="Line and zone preview (red = counting line, yellow = zone)", interactive=False)
        with gr.Row():
            out_video = gr.Video(label="Annotated result")
            with gr.Column():
                table = gr.Dataframe(headers=["Class", "Toward camera", "Away from camera", "Total"],
                                     interactive=False, label="Counts")
                summary = gr.Markdown()
        for comp in (video, line_y, roi_top):
            comp.change(preview_fn, [video, line_y, roi_top], preview)
        button.click(count_fn, [video, line_y, roi_top, direction, conf], [out_video, table, summary])
    return demo
