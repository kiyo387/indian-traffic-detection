# indian-traffic-detection

End-to-end computer vision pipeline for mixed Indian traffic: data prep, model comparison, tracking, and FastAPI deployment.

**Problem statement:** detect and count vehicles on Indian roads. Classes: car, motorcycle, rider, autorickshaw, bus, truck, person. Metric: mAP@50:95. Target speed: 20+ FPS.

## Dataset

[IDD Detection](http://idd.insaan.iiit.ac.in/) — 41,794 images, 507,576 bounding boxes across 15 classes (filtered to 7 road-user classes).

| Split | Images | Boxes | Notes |
| --- | ---: | ---: | --- |
| train | 6,082 | 95,911 | `highquality_16k` + `frontNear`, every 3rd frame |
| val | 5,138 | 67,298 | `highquality_16k` + `frontNear`, every 3rd frame |
| shift | 3,922 | 18,760 | `sideLeft`, `sideRight`, `rearNear`, `frontFar` — different viewpoints |

The shift set measures viewpoint **and** object scale together (side/rear cameras see different distributions of object sizes than front cameras), so shift-set numbers are not a pure viewpoint-robustness test.

## Results

| Run | val mAP@50:95 | shift mAP@50:95 | Notes |
| --- | ---: | ---: | --- |
| yolov8n, 30 epochs, 640 | 0.345 | 0.484 | baseline |

Shift mAP is higher than val mAP because the shift cameras (side/rear) see fewer tiny distant objects — the dominant failure mode on val.

## Failure Analysis

Full analysis with tables: [`results/failure_analysis.md`](results/failure_analysis.md).

**Summary:** The model's overall hit rate is 45.7%, with 51.1% misses and 3.1% wrong-class detections (at conf=0.25).

**Failure mode 1 — Tiny distant objects (dominant).** One-third of ground truth boxes are under 0.05% of the image area; the model finds only 7.3% of them, vs 93.6% for boxes over 2%. Boxes under 0.1% account for 76.2% of all misses. A `conf=0.1` rerun confirms the model genuinely cannot see them (recall rises only to 12.8%). Crowding amplifies this effect on tiny objects but has no impact on boxes ≥0.5%.

**Failure mode 2 — Bus/truck class confusion.** `bus` and `truck` have 8% wrong-class rate each and ~20pp lower recall than `car`/`motorcycle` at the same size — a real per-class difficulty, not a size artifact.

**Failure mode 3 — Apparent label noise (resolved).** Tiny GT boxes on building upper floors turned out to be real people on balconies at full resolution. No model fix needed.

> **Note on metrics:** For vehicle **counting**, objects a few pixels wide far down the road may not matter. Reporting mAP on boxes ≥0.1% of the image alongside overall mAP may tell a better story.

> **Note on selection bias:** The 20 failure gallery images are the highest-error images by construction. The quantitative tables (67,300 boxes) are the real prevalence measures, not per-image tallies.

## Week 2 Experiments

One change per run so each result is attributable:

| Experiment | Model | Epochs | imgsz | Hypothesis |
| --- | --- | ---: | ---: | --- |
| **A** | yolov8n | 80 | 640 | Baseline curves were still rising; isolate effect of training longer |
| **B** | yolov8n | 30 | 1280 | Tests the tiny-object hypothesis (~4× compute; `batch=4-8` for 6 GB GPU) |
| **C** | yolov8s | 30 | 640 | Tests model capacity; should reduce bus/truck confusion |
| **D** | RT-DETR | TBD | TBD | After A–C show which settings matter |

## Project Structure

```
configs/          YOLO dataset configs (idd.yaml, idd_shift.yaml)
notebooks/        EDA and analysis (01_eda.ipynb)
results/          Evaluation JSONs, parquet files, failure_analysis.md, failure gallery
runs/             YOLO training outputs
src/              Scripts (evaluate.py)
```
