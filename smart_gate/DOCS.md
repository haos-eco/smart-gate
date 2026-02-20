<img src="https://raw.githubusercontent.com/haos-eco/smart-gate/main/smart_gate/logo.png" alt="Smart Gate" />

# Smart Gate - License Plate Recognition

Automatic gate AI powered opener using YOLO + EasyOCR for license plate recognition.

**Author**: [haz3](https://github.com/andreaemmanuele)

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg

## Features
- 🚗 Automatic license plate detection using YOLOv11
- 🔍 OCR with EasyOCR for Italian plates (AA123AA format)
- 🤖 AI Super-Resolution (EDSR 2x) for enhanced accuracy on low-res crops
- 🌙 Automatic IR detection and exposure correction for night shots
- 🎯 Multi-attempt recognition — exact match takes priority, then best result selected by combined YOLO + OCR score
- 🔎 Fuzzy matching for OCR errors (up to 2 character tolerance)
- 📍 GPS-based security — fuzzy matches require the plate owner to be in home zone
- 🔒 Per-plate person entity — each plate linked to its specific owner
- 📊 Debug mode with per-attempt snapshot history

## Configuration

### Required Settings

- **Motion Entity**: Motion sensor that triggers when vehicle approaches
    - Example: `binary_sensor.driveway_motion`

- **Camera Entity**: High-resolution camera pointing at entrance
    - Recommended: 1080p or higher
    - Example: `camera.entrance_hd`

- **Gate Switch**: Switch entity to control gate
    - Example: `switch.main_gate`

- **Allowed Plates**: List of authorized license plates
    - Format: Italian plates (2 letters, 3 numbers, 2 letters)
    - Example: `['GT234YY', ...]`

### Optional Settings

- **Confidence**: YOLO detection threshold (default: 0.35)
    - Lower = more detections, more false positives
    - Higher = fewer detections, more accuracy

- **ROI (Region of Interest)**: Focus detection on specific area
    - Format: `x,y,width,height` (0.0-1.0 relative coordinates)
    - Default: `0.0,0.0,1.0,1.0` (full image)
    - Example: `0.0,0.3,1.0,0.4` (middle horizontal strip)

- **Cooldown**: Seconds to wait before next opening (default: 120)

- **Debug Mode**: Enable detailed logging and save crops
    - Creates debug images in `/config/www/smart_gate/snapshot/debug/`
    - Access via: `http://homeassistant.local:8123/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg`

- **Keep History**: Save all detection attempts with timestamps

## Setup

### 1. Install Addon

Add this repository to Home Assistant:
```
https://github.com/haos-eco/smart-gate
```

### 2. Download YOLO Model

You need to download a YOLOv11 license plate detection model:

**Option A: Pre-trained model (recommended)**
1. Download from: https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/best.onnx
2. Rename to `model.onnx`

**Option B: Train your own**
Train a custom YOLOv11 model on your specific plates/cameras

### 3. Upload Model

Upload `model.onnx` to Home Assistant:

**Via File Editor addon:**
1. Create folder (it should be created automatically): `/config/www/smart_gate/models/yolo/`
2. Upload `model.onnx` there

**Via Samba/SSH:**
```bash
scp model.onnx root@HOMEASSISTANT_IP:/config/www/smart_gate/models/yolo/
```

**Via WebUI:**
1. Install "File Editor" addon
2. Create `/config/www/smart_gate/models/yolo/` folder
3. Upload file

### 4. Configure

Add plates in configuration tab and adjust other fields based on your needs

### 5. Start Addon

First start will take 2-3 minutes to:
- Download AI super-resolution model (~1.5MB)
- Load EasyOCR models

### 6. Test (optional)

Enable debug mode and approach the gate. Check logs:
```
Motion detected! Multi-attempt recognition...
  Attempt 1/3: 'GT234YY' (YOLO score: 0.645)
  Attempt 2/3: 'GT234YY' (YOLO score: 0.658)
  Attempt 3/3: 'CT234YY' (YOLO score: 0.652)
📊 Consensus: 'GT234YY' (2/3 attempts)
✅ Gate opened for: GT234YY
```

## Troubleshooting

### No plates detected

1. Enable debug mode
2. Check debug image: `/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg`
3. Adjust ROI to focus on plate area
4. Lower confidence threshold (try 0.25)

### Wrong plate detected

1. Check lighting - add illumination if too dark
2. Verify camera angle - plate should be clearly visible
3. Check debug logs for OCR results
4. Multi-attempt consensus will help reduce errors

### Gate opens for wrong vehicles

1. Verify allowed_plates list is correct
2. Check if plates are similar (e.g., `GT234YY` vs `CT234YY`)
3. Increase confidence threshold
4. Adjust ROI to exclude other vehicles

## Performance

- **CPU**: 2-3 seconds per attempt (3 attempts = ~7 seconds total)
- **GPU**: Not required (CPU-only operation)
- **Memory**: ~500MB RAM
- **Storage**: ~200MB for models

## Privacy

All processing is done locally on your Home Assistant instance. No data is sent to external services.

## Support

For issues and feature requests:
https://github.com/haos-eco/smart-gate/issues

## Credits & Attribution

This addon uses the following open source components:

### YOLOv11 License Plate Detection Model
- **Model**: [yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
- **Author**: [MorseTechLab](https://huggingface.co/morsetechlab)
- **Base Framework**: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- **Training Platform**: [Roboflow](https://roboflow.com)
- **License**: GNU AGPLv3

### Libraries
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Apache License 2.0
- [OpenCV](https://opencv.org) - Apache License 2.0
- [ONNX Runtime](https://onnxruntime.ai) - MIT License

## License

This project is licensed under **GNU AGPLv3** (inherited from the YOLO model dependency).

See [LICENSE](../LICENSE) file for details.
