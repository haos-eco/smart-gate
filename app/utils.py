import os
import json

def get_options():
    """Load options from Home Assistant supervisor"""
    with open("/data/options.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
