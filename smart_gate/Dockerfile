FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    onnxruntime==1.16.3 \
    opencv-contrib-python==4.13.0.92 \
    pillow==12.1.1 \
    requests~=2.32.5 \
    easyocr~=1.7.2 \
    numpy~=2.4.2

WORKDIR /app

COPY app/ /app/
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD ["/run.sh"]
