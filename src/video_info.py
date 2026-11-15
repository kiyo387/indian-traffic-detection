import cv2, math
from pathlib import Path
import numpy as np

VID = Path(r"D:\DATA\traffic_videos")
files = sorted(VID.glob("*.*"), key=lambda p: (len(p.stem), p.stem))
files = [f for f in files if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]

rows, thumbs = [], []
for f in files:
    cap = cv2.VideoCapture(str(f))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, fr = cap.read()
    rows.append((f.name, f"{w}x{h}", round(fps, 1), n, round(n / fps, 1) if fps else 0))
    t = cv2.resize(fr, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8)
    cv2.putText(t, f.stem, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    thumbs.append(t); cap.release()

print(f"{'file':<14}{'size':<12}{'fps':<7}{'frames':<9}{'sec'}")
for r in rows:
    print(f"{r[0]:<14}{r[1]:<12}{r[2]:<7}{r[3]:<9}{r[4]}")

cols = 6; rws = math.ceil(len(thumbs) / cols)
thumbs += [np.zeros((180, 320, 3), np.uint8)] * (rws * cols - len(thumbs))
sheet = np.vstack([np.hstack(thumbs[i * cols:(i + 1) * cols]) for i in range(rws)])
Path("results").mkdir(exist_ok=True)
cv2.imwrite("results/video_contact_sheet.jpg", sheet)
print("saved results/video_contact_sheet.jpg")