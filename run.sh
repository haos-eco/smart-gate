#!/bin/bash
set -e

echo "Smart Gate v1 starting..."
echo "Python version:"
python3 --version

echo "Starting main.py with full output..."
python3 -u /app/main.py 2>&1
