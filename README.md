# indian-traffic-detection

End-to-end computer vision pipeline for mixed Indian traffic: dataset filtering, model comparison, failure analysis, FPS benchmarking, ByteTrack multi-object tracking, and directional vehicle counting with Hungarian evaluation.

**Problem statement:** Detect, track, and count vehicles across mixed Indian road conditions.
- **Classes (7):** `car`, `motorcycle`, `rider`, `autorickshaw`, `bus`, `truck`, `person` (counting excludes `rider`/`person` to prevent double-counting).
- **Core Metric:** mAP@50:95.
- **Speed Target:** 20+ FPS end-to-end on 1080p video.

---

## Dataset

Built from [IDD Detection](http://idd.insaan.iiit.ac.in/) (41,794 images, 507,576 bounding boxes across 15 classes, filtered to 7 road-user classes).

| Split | Images | Boxes | Notes |
| --- | ---: | ---: | --- |
| **train** | 6,082 | 95,911 | `highquality_16k` + `frontNear`, subsampled every 3rd frame |
| **val** | 5,138 | 67,298 | `highquality_16k` + `frontNear`, subsampled every 3rd frame |
| **shift** | 3,922 | 18,760 | `sideLeft`, `sideRight`, `rearNear`, `frontFar` (out-of-distribution viewpoints) |

*Known limitation:* Dropping `vehicle fallback` (~21k boxes), `caravan`, and `trailer` left those vehicles unlabeled, training models to treat them as background. Shift cameras also observe different object scale distributions than front cameras.

---

## Model Exploration & Results

Four experimental runs evaluated on identical `val` and `shift` sets:

| Run | Model | Config | val mAP@50:95 | shift mAP@50:95 | Speed (ms) | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **Baseline** | YOLOv8n | 30 ep, 640 | 0.345 | 0.484 | 4.6 ms | Baseline detector |
| **Test-time 1280** | YOLOv8n | 30 ep, test @ 1280 | 0.407 | 0.422 | 11.3 ms | No retrain; large vehicles degrade on shift |
| **Run A** | YOLOv8n | 80 ep, 640 | 0.354 | 0.488 | 4.6 ms | +0.9pp; proves 30 ep was not severely undertrained |
| **Run C** | YOLOv8s | 30 ep, 640 | 0.384 | 0.517 | 6.9 ms | +3.9pp; extra capacity helps class confusion |
| **Run B (Winner)** | YOLOv8n | 30 ep, 1280 | **0.443** | **0.528** | 11.1 ms | **+9.8pp** val mAP; recovers small/distant objects |

### Per-Class mAP@50:95 (`val`)

| Class | Baseline (640) | Run A (ep80) | Run C (yolov8s) | Run B (`imgsz=1280`) | Δ (vs Baseline) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `car` | 0.399 | 0.408 | 0.437 | **0.502** | +10.3pp |
| `motorcycle` | 0.310 | 0.318 | 0.346 | **0.411** | +10.1pp |
| `rider` | 0.250 | 0.259 | 0.282 | **0.347** | +9.7pp |
| `autorickshaw` | 0.408 | 0.418 | 0.453 | **0.513** | +10.5pp |
| `bus` | 0.464 | 0.478 | 0.502 | **0.535** | +7.1pp |
| `truck` | 0.388 | 0.395 | 0.431 | **0.475** | +8.7pp |
| `person` | 0.197 | 0.204 | 0.236 | **0.320** | +12.3pp |
| **Overall** | **0.345** | **0.354** | **0.384** | **0.443** | **+9.8pp** |

### Recall by Size Bin (`val`, `conf=0.25`)

| Size bin | Boxes in Val | Baseline (`640`) | Run B (`1280`) | Δ |
| --- | ---: | ---: | ---: | ---: |
| `<0.05%` area | 22,385 | 7.3% | 25.3% | **+18.0pp** |
| `0.05-0.1%` area | 8,905 | 30.9% | 56.9% | **+26.0pp** |
| `0.1-0.5%` area | 18,093 | 59.2% | 73.2% | **+14.0pp** |
| `0.5-2%` area | 9,608 | 82.4% | 86.7% | +4.3pp |
| `>2%` area | 8,309 | 93.6% | 94.2% | +0.6pp |

Tiny distant objects (~11×11 px at 640 vs ~21×21 px at 1280) were the primary failure mode of baseline detection. Run B sharpens recall on small objects without degrading larger objects.

Full quantitative breakdown and gallery: [`results/failure_analysis.md`](results/failure_analysis.md).

---

## Speed Benchmark & Model Choice

Evaluated on 1080p traffic video copies (`results/fps_benchmark_1080.csv`, timed across 200 frames, batch=1, FP32 GPU):

| Model | Resolution | Clip | Model FPS | End-to-End FPS (incl. decode) | Detections / Frame |
| --- | --- | --- | ---: | ---: | ---: |
| **YOLOv8n @ 1280** | 1920×1080 | test4 | 44.6 | 38.7 | 28.8 |
| YOLOv8s @ 640 | 1920×1080 | test4 | 52.3 | 44.7 | 14.6 |
| **YOLOv8n @ 1280** | 1920×1080 | test24 (dense) | 46.9 | 40.7 | 137.2 |
| YOLOv8s @ 640 | 1920×1080 | test24 (dense) | 53.6 | 45.9 | 73.2 |
| **YOLOv8n @ 1280** | 1920×1080 | test6 | 46.0 | 40.1 | 54.6 |
| YOLOv8s @ 640 | 1920×1080 | test6 | 52.6 | 44.6 | 37.3 |

**Selected Model:** `YOLOv8n @ 1280` (`models/final.pt`).
- Delivers **~40 FPS end-to-end** on 1080p, doubling the 20 FPS real-time target.
- Higher accuracy than YOLOv8s at 640 (+5.9pp val mAP, +1.1pp shift mAP).
- Detects nearly 2× more valid vehicles per frame in crowded Indian road scenes.

---

## Vehicle Tracking & Counting Pipeline

A complete directional counting system with polygon gating, track persistence, and anti-flicker guards:

```
Video Stream ──> YOLOv8n @ 1280 ──> ByteTrack ──> Bottom-Center Road Point
                                                         │
                                               [ROI Polygon Filter]
                                                         │
                                        [Analytical Segment Crossing (+/-)]
                                                         │
                 ┌───────────────────────────────────────┴───────────────────────────────────────┐
                 │ Stability Guards:                                                            │
                 │ 1. Minimum track age (≥5 frames inside ROI)                                  │
                 │ 2. Jump clamp (≤150 px frame-to-frame movement)                              │
                 │ 3. Lookback velocity check (≥5 px along line normal over 8 frames)           │
                 │ 4. Spatio-temporal deduplication (IoU ≥ 0.4 within 12 frames)                │
                 │ 5. Lifetime majority voting for vehicle class                                 │
                 │ 6. Class-agnostic NMS to suppress rider/two-wheeler double boxes             │
                 └───────────────────────────────────────┬───────────────────────────────────────┘
                                                         │
                                  ┌──────────────────────┴──────────────────────┐
                                  ▼                                             ▼
                      count_<clip>.mp4 (HUD overlay)               count_<clip>_events.csv
                      count_<clip>_review.jpg (audit crop sheet)
```

### Pipeline Tools

1. **Interactive Geometry Picker (`src/pick_geometry.py`):** Click counting line (start/end defining normal direction) and arbitrary polygon ROI zone. Saves geometry JSON and overlay PNG.
2. **Traffic Counter (`src/count_traffic.py`):** Runs detection, ByteTrack tracking, crossing verification, deduplication, and renders live HUD counts.
3. **Contact Sheet Generator:** Automatically crops counted vehicles into a grid image (`_review.jpg`) with timestamp, track ID, and direction for quick visual audit.

---

## Counting Evaluation

Line crossings are evaluated against manual frame-by-frame ground truth using the **Hungarian algorithm** (`src/eval_counts.py`). Pairs are matched by time proximity (`--tol 0.3` seconds) and movement direction, with an extra penalty for mismatched classes.

Evaluation is split into **"tuned on"** clips (where geometry and filters were calibrated during development) and **"not tuned on"** clips (blind evaluation including fresh footage):

| Group | Clip | Window | Manual | Detected | Matched | Missed | Extra | Class Accuracy | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tuned on** | `test17` | 10–20s | 18 | 18 | 16 | 2 | 2 | 15/16 (94%) | Overhead view, northbound |
| **Tuned on** | `test4` | 5–15s | 20 | 15 | 15 | 5 | 0 | 15/15 (100%) | Mixed arterial road |
| **Tuned on** | `test10` | 10–20s | 22 | 18 | 17 | 5 | 1 | 17/17 (100%) | Highway flyover |
| **Not tuned on** | `test6` | 1–6s | 53 | 43 | 41 | 12 | 2 | 39/41 (95%) | High-volume two-way arterial |
| **Not tuned on** | `test24` | 2–7s | 8 | 5 | 5 | 3 | 0 | 5/5 (100%) | Standstill dense jam stress test |
| **Not tuned on (Fresh)** | `test16` | 1–9s | 12 | 13 | 12 | 0 | 1 | 12/12 (100%) | Fresh blind test |
| **Not tuned on (Fresh)** | `test25` | 10–20s | 21 | 31 | 21 | 0 | 10 | 20/21 (95%) | Fresh blind test |

### Known Failure Modes & Limitations
1. **Missed Crossings:**
   - **Occluded two-wheelers:** In dense traffic, motorcycles riding directly between or behind cars are heavily occluded from camera view (5 of 11 misses on earlier clips). Lowering detector confidence does not recover them without introducing false detections.
   - **ID switches on tailgating vehicles:** In closely following traffic, ByteTrack may latch an active track ID from a leading vehicle directly onto the trailing vehicle.
   - **Standstill congestion:** In heavy stop-and-go jams (e.g. `test24`), static vehicles with micro-jitter can cause tracker fragmentation or false reverse-direction tracks.
2. **Per-Class Reliability:**
   - Per-class counts are less reliable for `autorickshaw` and `truck` due to viewpoint ambiguity and class confusion with cars/mini-trucks.
3. **Direction Constraint (`--only_dir`):**
   - `--only_dir` (`to_plus` or `to_minus`) is a per-scene operational configuration for known one-way road segments. It suppresses wrong-way tracks by construction. On two-way streets (`test6`, `test16`, `test25`), directional counting runs unconstrained.

---

## Project Structure

```
configs/
  idd.yaml                    # IDD dataset train/val config
  idd_shift.yaml              # Shift-set evaluation config
  geometry_test*.json         # Counting line and ROI definitions
  geometry_test*.png          # Visual verification overlays
models/
  final.pt                    # Selected model checkpoint (YOLOv8n @ 1280)
notebooks/
  01_eda.ipynb                # Dataset exploratory data analysis
results/
  *.json                      # Model evaluation metrics (mAP, speed)
  *.parquet                   # Match tables for failure analysis
  failure_analysis.md         # Deep-dive error analysis
  fps_benchmark_1080.csv      # Real-time latency benchmark on 1080p clips
  count_*.mp4                 # Annotated video outputs with counts HUD
  count_*_events.csv          # Event-level crossing log
  count_*_review.jpg          # Cropped vehicle audit contact sheets
src/
  benchmark_all.py            # Latency benchmark harness across clips & models
  benchmark_fps.py            # Single-run FPS timer
  count_traffic.py            # Real-time tracking and directional counting pipeline
  error_analysis.py           # Generates GT match tables and recall by bin/class
  eval_counts.py              # Hungarian matching evaluation against manual counts
  evaluate.py                 # Evaluates mAP on YOLO validation sets
  pick_geometry.py            # Interactive OpenCV GUI for line and ROI calibration
  track_video.py              # Basic ByteTrack video visualizer
  video_info.py               # Video metadata probe and contact sheet creator
```

---

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows (.venv/bin/activate on Linux)
pip install -r requirements.txt  # ultralytics opencv-python pandas scipy
```

### 2. Calibrate Line & ROI on a Video

```bash
python src/pick_geometry.py --video path/to/video.mp4 --frame 300 --out configs/geometry_myvideo.json
```
- **Step 1:** Click two points to define the counting line (first point to second point defines the vector, normal defines `+` vs `-`). Press `Enter`.
- **Step 2:** Click 3+ points to define the tracking ROI zone. Press `Enter`.

### 3. Run Tracking and Vehicle Counting

```bash
python src/count_traffic.py \
  --video path/to/video.mp4 \
  --weights models/final.pt \
  --geometry configs/geometry_myvideo.json \
  --imgsz 1280 \
  --conf 0.25
```
Outputs written to `results/`:
- `count_<video>.mp4` (video with HUD and tracks)
- `count_<video>_events.csv` (time, track ID, class, direction)
- `count_<video>_review.jpg` (visual crop contact sheet)

### 4. Evaluate Counting Accuracy Against Manual Counts

```bash
python src/eval_counts.py \
  --events results/count_test17_v2_events.csv \
  --manual data/manual_test17.csv \
  --window 10 20 \
  --tol 1.0
```

| clip | window | manual | detected | matched | missed | extra | class_correct |
|---|---|---|---|---|---|---|---|
| test17 | 10-20s | 18 | 17 | 16 | 2 | 1 | 15/16 |
| test4 | 5-15s | 20 | 15 | 15 | 5 | 0 | 15/15 |
| test10 | 10-20s | 22 | 18 | 18 | 4 | 0 | 17/18 |
