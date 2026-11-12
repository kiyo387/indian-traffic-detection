import argparse
import numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"D:/DATA/idd_yolo")
BINS = [0, 0.0005, 0.001, 0.005, 0.02, 1]
LABELS = ["<0.05%", "0.05-0.1%", "0.1-0.5%", "0.5-2%", ">2%"]


def to_xyxy(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="val", choices=["val", "shift"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    model = YOLO(a.weights)
    names = model.names
    results = model.predict(str(ROOT / "images" / a.split), stream=True, conf=a.conf,
                            imgsz=a.imgsz, device=0, verbose=False, batch=a.batch)

    gt_rows, img_rows = [], []
    for n, r in enumerate(results):
        stem = Path(r.path).stem
        lab = ROOT / "labels" / a.split / (stem + ".txt")
        gts = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        gts = [(int(g[0]), to_xyxy(*map(float, g[1:]))) for g in gts]
        preds = [(int(c), b.cpu().numpy(), float(s)) for c, b, s in
                 zip(r.boxes.cls, r.boxes.xyxyn, r.boxes.conf)]
        preds.sort(key=lambda t: -t[2])

        used, n_miss = set(), 0
        for gc, gb in gts:
            best, bi = 0, -1
            for i, (pc, pb, ps) in enumerate(preds):
                if i in used:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, i
            area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            if best >= 0.5:
                used.add(bi)
                outcome = "hit" if preds[bi][0] == gc else "wrong_class"
            else:
                outcome, n_miss = "miss", n_miss + 1
            gt_rows.append({"img": stem, "cls": names[gc], "area": area, "outcome": outcome})
        img_rows.append({"img": stem, "gts": len(gts), "misses": n_miss, "fp": len(preds) - len(used)})
        if n % 500 == 0:
            print(n, "images done", flush=True)

    gt, per_img = pd.DataFrame(gt_rows), pd.DataFrame(img_rows)
    out = Path("results"); out.mkdir(exist_ok=True)
    gt.to_parquet(out / f"{a.name}_gt_matches.parquet")
    per_img.to_parquet(out / f"{a.name}_per_image.parquet")

    gt["size"] = pd.cut(gt.area, BINS, labels=LABELS)
    print(f"\n{a.name} ({a.split}, imgsz={a.imgsz}, conf={a.conf})")
    print("recall by size bin:")
    print(gt.groupby("size", observed=True).outcome.apply(lambda s: (s == "hit").mean()).round(3))
    print("recall by class:")
    print(gt.groupby("cls").outcome.apply(lambda s: (s == "hit").mean()).round(3))


if __name__ == "__main__":
    main()