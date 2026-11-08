import argparse, json
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="configs/idd.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", required=True)          # label for the results file
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    m = YOLO(a.weights).val(data=a.data, imgsz=a.imgsz, batch=a.batch,
                            workers=a.workers, device=0, split="val", plots=False)

    res = {
        "name": a.name, "weights": a.weights, "data": a.data, "imgsz": a.imgsz,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class_map50_95": {m.names[int(i)]: float(v)
                               for i, v in zip(m.box.ap_class_index,
                                               m.box.maps[m.box.ap_class_index])},
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in m.speed.items()},
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()