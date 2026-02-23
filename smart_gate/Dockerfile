FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU-only — heaviest layer, changes rarely (~800MB)
RUN pip install --no-cache-dir \
    torch~=2.4.0 --index-url https://download.pytorch.org/whl/cpu

# ML dependencies — changes occasionally
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    easyocr~=1.7.2 \
    transformers~=4.44.0

# Light dependencies — change frequently
RUN pip install --no-cache-dir \
    onnxruntime==1.18.0 \
    opencv-contrib-python-headless==4.10.0.84 \
    pillow==10.4.0 \
    requests~=2.32.5

WORKDIR /app
COPY app/ /app/
COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
