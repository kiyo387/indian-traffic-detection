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

Boxes under 0.1% of the image account for 76.2% of all misses (26,225 of 34,416), and 93.9% of all incorrectly matched boxes (34,302 of 36,530) are under 0.5%.

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

`person`'s terrible overall recall (0.31) is mostly a size effect: 57.4% of person boxes are under 0.1% of the image, though a modest gap vs `car` remains from 0.1% up (`0.56` vs `0.65`, `0.82` vs `0.88`, `0.86` vs `0.97`). The biggest outliers are **`bus` and `truck`**, which have ~20pp lower recall than `car`/`motorcycle`/`autorickshaw` at 0.1–0.5% and 0.5–2% (`0.37–0.38` and `0.67` vs `0.58–0.65` and `0.87–0.89`), with ≥303 boxes in every cell (`wrong_class` is 8% for both). Possible factors include `bus`/`truck` mutual confusion, truncation/occlusion on large vehicles, and dropped classes (`vehicle fallback` ~21k boxes, `caravan`, `trailer` left unlabeled as background).

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

**Failure mode 1: small objects (dominant)**
Evidence: 33.3% of ground-truth boxes (22,385) are under 0.05% of the image area and are found 7.3% of the time, against 82–94% for boxes over 0.5%. 93.9% of all incorrectly matched boxes (34,302 of 36,530) are under 0.5%. The `conf=0.1` check confirms the model genuinely cannot see these objects (recall rises only to 12.8% for `<0.05%` while `wrong_class` doubles 3.1% → 6.3%).
Examples: `results/failures/highquality_16k__15-07-18-upload__2018-05-08_4-20-0-775835_leftImg8bit.jpg`, `results/failures/highquality_16k__HYD-2018-04-26_12-40-01__0005609.jpg`
Hypothesis: a larger input size (`imgsz` 960 or 1280) recovers part of these, though not all, since a 0.05% box is ~11×11 pixels at 640 and ~21×21 at 1280. Alternatively, reporting metrics on boxes ≥0.1% alongside overall mAP better reflects vehicle counting where distant horizon pixels do not matter.

**Failure mode 2: crowding amplifies small-object misses**
Evidence: within the `<0.05%` bin recall falls from 0.20 (1–5 boxes) to 0.04 (40+ boxes), and within `0.05-0.1%` from 0.48 to 0.23. Above 0.5% there is no effect (`0.81–0.83` and `0.93–0.94` across all crowd levels). Note that the hand-sorted tally comes from the 20 worst images in `results/failures/` (all `highquality_16k`), so "15/15 contain tiny objects" reflects selection, not prevalence.
Examples: `results/failures/highquality_16k__BLR-2018-05-31_10-49-32__2018-05-31_10-57-56-135029_leftImg8bit.jpg`, `results/failures/highquality_16k__15-07-18-upload__2018-05-08_5-3-32-770267_leftImg8bit.jpg`
Hypothesis: higher resolution (`imgsz=1280`) and a larger model/tuned NMS will reduce small-object collapse in dense scenes, though within-bin size variation and box overlap among tiny objects may leave some gap.

**Failure mode 3: class-specific gaps**
Evidence: at 0.5–2% area, `bus` and `truck` are found 67% of the time vs 87–89% for `car`, `motorcycle`, and `autorickshaw` (and 0.37–0.38 vs 0.58–0.65 at 0.1–0.5%, with ≥303 boxes per cell). `person` is lower than `car` at every size from 0.1% up (`0.56` vs `0.65`, `0.82` vs `0.88`, `0.86` vs `0.97`). Secondary issues in the hand-sorted images: two-wheeler plus rider predicted as `autorickshaw` (4 of 15 sorted images) and shadowed scenes (3 of 15). Zoomed crops of upper-floor balcony boxes (`highquality_16k__BLR-2018-06-19-06-21-44_part_47__0001013.jpg`, boxes 28–29) showed real people rather than label noise.
Examples: `results/failures/highquality_16k__HYD-2018-04-26_13-40-01__0007741.jpg`, `results/failures/highquality_16k__HYD-2018-04-26_13-40-01__0006928.jpg`, `results/failures/highquality_16k__BLR-2018-06-20-05-53-14_part_4__0001049.jpg`
Hypothesis: `yolov8s` helps with capacity-related confusion (`autorickshaw` vs two-wheeler, `person`); the cause for `bus` and `truck` still needs checking the confusion matrix alongside truncation and unlabeled `vehicle fallback` objects.
