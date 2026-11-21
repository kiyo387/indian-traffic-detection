FROM python:3.12-slim

# ffmpeg: re-encode uploads and the annotated result; libgl1 and libglib2.0-0: needed by OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch first, so installing ultralytics does not pull the multi-gigabyte CUDA build
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user app ./app
COPY --chown=user src/count_traffic.py ./src/count_traffic.py
COPY --chown=user models/final.pt ./models/final.pt

ENV DEVICE=cpu \
    IMGSZ=1280 \
    STRIDE=2 \
    MAX_SECONDS=10 \
    MAX_UPLOAD_MB=60 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PYTHONUNBUFFERED=1

USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
