# Failure Analysis

Source data: `results/baseline_gt_matches.parquet` (67,300 GT boxes across 5,138 val images, matched at `conf=0.25`) and 15 hand-sorted images from `results/failures/`.

> [!NOTE]
> The 20 images in `results/failures/` are the **highest-error** images by construction (top-20 by misses + false positives). All 15 sorted images come from `highquality_16k`, which has the most crowded, distant scenes. So per-image tallies like "15/15 have tiny objects" reflect the selection, not the dataset. The quantitative tables below (67,300 boxes) are the real prevalence measures.

## Quantitative Tables

### Recall by size bin

| Size bin | Boxes | % of GT | Found (`hit`) |
| --- | ---: | ---: | ---: |
| `<0.05%` | 22,385 | 33.3% | 7.3% |
| `0.05-0.1%` | 8,905 | 13.2% | 30.9% |
| `0.1-0.5%` | 18,093 | 26.9% | 59.2% |
| `0.5-2%` | 9,608 | 14.3% | 82.4% |
| `>2%` | 8,309 | 12.3% | 93.6% |

Boxes under 0.1% of the image account for 76.2% of all misses (26,225 of 34,416).

### Recall by crowding × size (is crowding an independent cause?)

| Size bin | 1–5 boxes | 6–15 | 16–40 | 40+ |
| --- | ---: | ---: | ---: | ---: |
| `<0.05%` | 0.20 | 0.11 | 0.06 | 0.04 |
| `0.05-0.1%` | 0.48 | 0.38 | 0.30 | 0.23 |
| `0.1-0.5%` | 0.68 | 0.62 | 0.58 | 0.57 |
| `0.5-2%` | 0.83 | 0.81 | 0.83 | 0.82 |
| `>2%` | 0.94 | 0.94 | 0.94 | 0.93 |

For boxes ≥0.5% of the image, recall is flat across crowd levels (~0.82–0.94). Crowding hurts only on tiny objects: recall on `<0.05%` boxes drops from 0.20 in sparse images to 0.04 in 40+ images. So **crowding is not an independent failure mode** for medium/large objects — it amplifies the size effect.

### Recall by class × size (is class an independent cause?)

| Class | `<0.05%` | `0.05-0.1%` | `0.1-0.5%` | `0.5-2%` | `>2%` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `autorickshaw` | 0.03 | 0.27 | 0.58 | 0.87 | 0.95 |
| `bus` | 0.01 | 0.09 | 0.38 | 0.67 | 0.90 |
| `car` | 0.07 | 0.37 | 0.65 | 0.88 | 0.97 |
| `motorcycle` | 0.13 | 0.41 | 0.65 | 0.89 | 0.95 |
| `person` | 0.02 | 0.20 | 0.56 | 0.82 | 0.86 |
| `rider` | 0.08 | 0.35 | 0.61 | 0.77 | 0.90 |
| `truck` | 0.02 | 0.14 | 0.37 | 0.67 | 0.90 |

`person`'s terrible overall recall (0.31) is mostly a size effect: 57.4% of person boxes are under 0.1% of the image. Within a size bin, most classes look similar. The exceptions are **`bus` and `truck`**, which have ~20pp lower recall than `car`/`motorcycle` at the same size — suggesting a real class-level difficulty (possibly because they share visual features and get confused with each other; `wrong_class` is 8% for both).

### Class crosstab (row-normalized)

| Class | `hit` | `miss` | `wrong_class` |
| --- | ---: | ---: | ---: |
| `autorickshaw` | 0.53 | 0.43 | 0.05 |
| `bus` | 0.54 | 0.38 | 0.08 |
| `car` | 0.54 | 0.44 | 0.02 |
| `motorcycle` | 0.51 | 0.48 | 0.02 |
| `person` | 0.31 | 0.66 | 0.03 |
| `rider` | 0.39 | 0.58 | 0.03 |
| `truck` | 0.48 | 0.44 | 0.08 |

### Confidence threshold check (conf=0.1 vs conf=0.25)

| Size bin | Recall @ 0.25 | Recall @ 0.10 | Δ |
| --- | ---: | ---: | ---: |
| `<0.05%` | 7.3% | 12.8% | +5.5pp |
| `0.05-0.1%` | 30.9% | 44.3% | +13.4pp |
| `0.1-0.5%` | 59.2% | 68.1% | +8.9pp |
| `0.5-2%` | 82.4% | 86.1% | +3.7pp |
| `>2%` | 93.6% | 94.7% | +1.1pp |

Overall: `hit` rose 45.7% → 52.4%, but `wrong_class` doubled 3.1% → 6.3%. The model has some low-confidence correct detections (especially for mid-size boxes), but tiny objects remain catastrophically low at 12.8%. Lowering the threshold is not a substitute for `imgsz` or model capacity improvements.

---

## Failure Modes

**Failure mode 1: tiny distant objects (dominant)**
Evidence: boxes under 0.05% of the image area (22,385 boxes, 33.3% of ground truth) are found just 7.3% of the time, vs 93.6% for boxes over 2%. Boxes under 0.1% account for 76.2% of all misses (26,225 of 34,416). Crowding amplifies this — the same tiny boxes drop from 20% recall in sparse images to 4% in crowded images — but for boxes ≥0.5%, crowding has no effect. The `conf=0.1` check confirms the model genuinely cannot see these objects (recall rises only to 12.8%).
Examples: `results/failures/highquality_16k__15-07-18-upload__2018-05-08_4-20-0-775835_leftImg8bit.jpg`, `results/failures/highquality_16k__HYD-2018-04-26_12-40-01__0005609.jpg`
Hypothesis: raising `imgsz` from 640 to 1280 will partially recover these. Under 0.05% of a 1920×1080 frame is ~32×32 px; at `imgsz=640` that shrinks to ~11×11 px, at 1280 to ~21×21 px. Expect partial, not full, recovery. Alternatively, reporting metrics on boxes ≥0.1% of the image may be a better story for vehicle counting, where objects a few pixels wide far down the road may not matter.

**Failure mode 2: bus/truck class confusion**
Evidence: `bus` and `truck` have 8% `wrong_class` rate each (vs 2–3% for other classes), and within the same size bins they have ~20pp lower recall than `car` or `motorcycle`. This is a real per-class difficulty, not a size artifact.
Examples: `results/failures/highquality_16k__HYD-2018-04-26_13-40-01__0007741.jpg`, `results/failures/highquality_16k__15-07-18-upload__2018-05-31_11-2-8-340904_leftImg8bit.jpg`
Hypothesis: a bigger model (`yolov8s`) with more capacity should reduce bus/truck confusion.

**Failure mode 3: possible label noise (minor)**
Evidence: some images have tiny ground-truth boxes on upper-floor building areas. On manual inspection (cropped at full resolution), these turned out to be **real people on balconies**, not labeling errors — so this is actually correct annotation of genuinely difficult objects, not label noise. Noted here to explain why 3/15 failure images appeared to have spurious labels.
Examples: `results/failures/highquality_16k__BLR-2018-06-19-06-21-44_part_47__0001013.jpg` (boxes 28, 29 — people on upper walkway)
Hypothesis: no model fix needed; these are real objects that happen to be very small (covered by failure mode 1).
