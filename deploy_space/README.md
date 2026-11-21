---
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
- Source code: https://github.com/kiyo387/indian-traffic-detection
