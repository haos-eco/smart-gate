#!/bin/bash
set -e

echo "Smart Gate v1 starting..."
echo "Python version:"
python3 --version

python -c "import cv2, onnxruntime, easyocr; print('imports OK')" || exit 1

echo "Starting main.py with full output..."
python3 -u /app/main.py 2>&1
