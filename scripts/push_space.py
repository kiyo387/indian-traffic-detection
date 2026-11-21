"""Build the Hugging Face Space folder (deploy_space/) and optionally upload it.

    python scripts/push_space.py                          # build only, then inspect deploy_space/
    python scripts/push_space.py --push USER/space-name   # build and upload (needs a write token)
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy_space"

SPACE_README = """---
title: Indian Traffic Detection and Counting
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Indian Traffic Detection and Counting

A YOLOv8 detector trained on the Indian Driving Dataset (IDD), with ByteTrack tracking and line-crossing vehicle counts.
Upload a short clip from a fixed camera, place the counting line, and get per-class counts and an annotated video.

- Demo limits: first 10 seconds of a clip, up to 60 MB, small CPU, one video at a time.
- Counts are approximate. See the project README for the evaluation and its limitations.
- REST API: open `/docs` on this Space for the interactive API documentation.
{source}"""


def build(github):
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    (OUT / "models").mkdir()
    shutil.copytree(ROOT / "app", OUT / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "src" / "count_traffic.py", OUT / "src" / "count_traffic.py")
    weights = ROOT / "models" / "final.pt"
    if not weights.exists():
        raise SystemExit(f"missing {weights}: copy your best.pt there first")
    shutil.copy(weights, OUT / "models" / "final.pt")
    for name in ("Dockerfile", "requirements-api.txt"):
        shutil.copy(ROOT / name, OUT / name)
    source = f"- Source code: {github}\n" if github else ""
    (OUT / "README.md").write_text(SPACE_README.format(source=source), encoding="utf-8")
    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    print("built", OUT)
    for f in files:
        print("  ", f, f"({(OUT / f).stat().st_size / 1e6:.1f} MB)" if f.endswith(".pt") else "")


def push(repo_id):
    from huggingface_hub import HfApi          # uses a saved login or the HF_TOKEN environment variable
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy traffic detection and counting demo")
    print(f"uploaded. Build logs and the app: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--push", metavar="USER/SPACE_NAME", default=None)
    p.add_argument("--github", default="", help="URL of the GitHub repository, shown in the Space README")
    a = p.parse_args()
    build(a.github)
    if a.push:
        push(a.push)
