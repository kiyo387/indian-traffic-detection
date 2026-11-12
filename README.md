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

**Known limitation:** Dropping `vehicle fallback` (~21k boxes), `caravan`, and `trailer` left those vehicles unlabeled in the images, so the model is trained to treat them as background.

## Results

| Run | val mAP@50:95 | shift mAP@50:95 | Notes |
| --- | ---: | ---: | --- |
| yolov8n, 30 epochs, 640 | 0.345 | 0.484 | baseline |

Shift mAP is higher than val mAP because the shift cameras (side/rear) see fewer tiny distant objects — the dominant failure mode on val.

## Failure Analysis

Full analysis with tables: [`results/failure_analysis.md`](results/failure_analysis.md).

**Summary:** At `conf=0.25`, overall GT hit rate is 45.7%, with 51.1% misses and 3.1% wrong-class detections.

**Failure mode 1 — Small objects (dominant).** 33.3% of ground-truth boxes are under 0.05% of the image area and are found 7.3% of the time, against 82–94% for boxes over 0.5%. 93.9% of all incorrectly matched boxes (34,302 of 36,530) are under 0.5%. A `conf=0.1` rerun confirms the model genuinely cannot see tiny boxes (recall rises only to 12.8%, while `wrong_class` doubles 3.1% → 6.3%). Hypothesis: raising `imgsz` to 960 or 1280 recovers part of these (~11×11 px at 640 vs ~21×21 px at 1280 for a 0.05% box), though not all.

**Failure mode 2 — Crowding amplifies small-object misses.** Within `<0.05%` recall falls from 0.20 (1–5 boxes) to 0.04 (40+ boxes), and within `0.05-0.1%` from 0.48 to 0.23. Above 0.5% there is no effect (`0.81–0.83` and `0.93–0.94`). Note that the hand-sorted tally comes from the 20 worst images in `results/failures/`, so "15/15 contain tiny objects" reflects selection, not prevalence.

**Failure mode 3 — Class-specific gaps.** At 0.5–2% area, `bus` and `truck` are found 67% of the time vs 87–89% for `car`, `motorcycle`, and `autorickshaw` (and 0.37–0.38 vs 0.58–0.65 at 0.1–0.5%, with ≥303 boxes per cell). `person` is lower than `car` at every size from 0.1% up (`0.56` vs `0.65`, `0.82` vs `0.88`, `0.86` vs `0.97`). Secondary issues: two-wheeler plus rider predicted as `autorickshaw` (4 of 15 sorted images) and shadowed scenes (3 of 15); zoomed crops of tiny upper-floor balcony boxes (`part_47__0001013`) showed real people rather than label noise. Hypothesis: `yolov8s` helps with capacity-related confusion; the cause for `bus` and `truck` still needs the confusion matrix.

> **Note on metrics:** For vehicle **counting**, objects a few pixels wide far down the road may not matter. Reporting mAP on boxes ≥0.1% of the image alongside overall mAP may tell a better story.

> **Note on selection bias:** The 20 failure gallery images are the highest-error images by construction. The quantitative tables (67,300 boxes) are the real prevalence measures, not per-image tallies.

## Week 2 Experiments

One change per run on the same `val` and `shift` sets, starting with **B**:

| Run | Change from baseline | Tests |
| --- | --- | --- |
| **B** | `imgsz=1280`, 30 epochs | Tiny-object hypothesis (~4× compute; `batch=4–8` for 6 GB GPU) |
| **A** | 80 epochs, `imgsz=640` | Undertrained curves (baseline curves were still rising) |
| **C** | `yolov8s`, 30 epochs | Model capacity / class confusion |
| **D** | RT-DETR, later | Detector family, once A–C show which settings matter |

## Project Structure

```
configs/          YOLO dataset configs (idd.yaml, idd_shift.yaml)
notebooks/        EDA and analysis (01_eda.ipynb)
results/          Evaluation JSONs, parquet files, failure_analysis.md, failure gallery
runs/             YOLO training outputs
src/              Scripts (evaluate.py)
```
