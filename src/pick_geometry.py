import argparse, json
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=600)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read that frame"
    h, w = frame.shape[:2]
    DW = 1280
    disp = cv2.resize(frame, (DW, int(h * DW / w)))
    DH = disp.shape[0]

    line_pts, roi_pts, stage = [], [], [1]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if stage[0] == 1 and len(line_pts) < 2:
                line_pts.append((x, y))
            elif stage[0] == 2:
                roi_pts.append((x, y))

    def render():
        img = disp.copy()
        if len(roi_pts) >= 2:
            cv2.polylines(img, [__import__("numpy").array(roi_pts, dtype="int32")], len(roi_pts) >= 3, (0, 200, 255), 2)
        for q in roi_pts:
            cv2.circle(img, q, 4, (0, 200, 255), -1)
        for q in line_pts:
            cv2.circle(img, q, 5, (0, 0, 255), -1)
        if len(line_pts) == 2:
            cv2.line(img, line_pts[0], line_pts[1], (0, 0, 255), 2)
        msg = ("STEP 1: click both ends of the counting line, then ENTER. R = reset" if stage[0] == 1 else
               "STEP 2: click corners of the counting zone (3+), then ENTER. R = reset")
        cv2.rectangle(img, (0, 0), (DW, 30), (0, 0, 0), -1)
        cv2.putText(img, msg, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    cv2.namedWindow("pick", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("pick", on_mouse)
    while True:
        cv2.imshow("pick", render())
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            print("cancelled, nothing saved")
            return
        if k in (ord("r"), ord("R")):
            (line_pts if stage[0] == 1 else roi_pts).clear()
        if k == 13:
            if stage[0] == 1 and len(line_pts) == 2:
                stage[0] = 2
            elif stage[0] == 2 and len(roi_pts) >= 3:
                break
    cv2.destroyAllWindows()

    geom = {"line": [[x / DW, y / DH] for x, y in line_pts], "roi": [[x / DW, y / DH] for x, y in roi_pts]}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(geom, indent=2))
    cv2.imwrite(str(out.with_suffix(".png")), render())
    print("saved", out)


if __name__ == "__main__":
    main()