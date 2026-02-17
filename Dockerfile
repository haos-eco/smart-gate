FROM python:3.11-slim-bullseye

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installa pacchetti Python
RUN pip install --no-cache-dir \
    onnxruntime \
    opencv-python-headless \
    pillow \
    requests \
    easyocr

WORKDIR /app

COPY app/ /app/
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD ["/run.sh"]
