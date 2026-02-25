<img src="https://raw.githubusercontent.com/haos-eco/smart-gate/main/smart_gate/logo.png" alt="Smart Gate" />

# Smart Gate - License Plate Recognition

Automatic gate AI powered opener using YOLO + TrOCR (EasyOCR as fallback) for license plate recognition.

**Author**: [haz3](https://github.com/andreaemmanuele)

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg

## Features
- 🚗 Automatic license plate detection using YOLOv11
- 🔍 OCR with TrOCR (primary) or EasyOCR (fallback) for Italian plates (AA123AA format)
- 🤖 AI Super-Resolution (EDSR 2x) for enhanced accuracy on low-res crops
- 🌙 Automatic IR detection and exposure correction for night shots
- 🎯 Multi-attempt recognition — exact whitelist match triggers early exit, otherwise best result selected by combined YOLO + OCR score
- 🔎 Fuzzy matching for OCR errors (up to 2 character tolerance)
- 📍 GPS-based security — fuzzy matches require the plate owner to be in home zone
- 🔒 Per-plate person entity — each plate linked to its specific owner
- 🔔 Visitor notifications — actionable push notification with annotated snapshot when an unknown vehicle stops
- 🖼️ Annotated snapshots — bounding box and OCR result drawn on every notification image
- 🚫 Privacy zones — mask areas of the camera frame before recognition and notifications
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
  - Simple format: `['GT234YY', ...]`
  - With GPS check (for fuzzy matches): `[{plate: "GT234YY", person_entity: "person.andrea"}, ...]`

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

- **Notify Services**: List of `notify.*` services to receive visitor notifications
  - Example: `['notify.mobile_app_iphone_di_andrea']`
  - See [Visitor Notifications](#visitor-notifications) section below

- **Visitor Stop Seconds** (`visitor_stop_sec`): How long the vehicle must be stopped before a visitor notification is sent (default: `5`)

- **Notification Sound** (`notification_sound`): iOS notification sound for visitor alerts (default: `default`)

- **Notify on Failure** (`notify_on_failure`): Send a notification with annotated snapshot when a vehicle is detected but the plate is unreadable, rejected, or not in whitelist (default: `false`)

- **Privacy Zones**: List of rectangular areas to black out before recognition and notifications. Defined in `config.yaml` and hot-reloadable without addon restart.
  - See [Privacy Zones](#privacy-zones) section below

## Visitor Notifications

When a vehicle is detected but does not trigger an automatic opening (unknown visitor, not in whitelist, or simply stopped at the gate), Smart Gate can send a push notification to your phone.

The notification includes:
- An **annotated snapshot** showing the detected license plate with a bounding box, the OCR reading, and YOLO/OCR confidence scores
- A **"🔓 Apri Cancello"** action button — long-press the notification on iOS to open the gate remotely without opening the app

### Setup

1. Install the **Home Assistant Companion App** on your device
2. Add your notify service to the addon configuration:
```yaml
notify_services:
  - notify.mobile_app_iphone_di_andrea
```
3. Optionally tune when the notification fires:
```yaml
visitor_stop_sec: 5       # send after vehicle has been stopped 5 seconds
notification_sound: default
notify_on_failure: true   # also notify for unreadable/rejected plates
```

### What triggers a notification

| Situation | Notification sent |
|---|---|
| Unknown visitor stopped at gate | ✅ Always (if `notify_services` set) |
| Plate unreadable | ✅ If `notify_on_failure: true` |
| Plate rejected (low confidence) | ✅ If `notify_on_failure: true` |
| Plate not in whitelist | ✅ If `notify_on_failure: true` |
| Fuzzy match but owner not home | ✅ If `notify_on_failure: true` |
| Authorized plate → gate opened | ❌ No notification |

## Privacy Zones

Privacy zones let you black out specific areas of the camera frame — for example, a public street visible in the background — before any recognition, logging, or notification occurs. Masked areas are also excluded from motion detection.

Zones are configured in `/data/config.yaml` and reloaded automatically every 5 seconds without restarting the addon.

### Define zones via the dashboard

Open the **Smart Gate dashboard → Zone Maschera tab**. Draw rectangles directly on a live camera frame, then copy the generated YAML.

### Manual configuration

Add to `/data/config.yaml`:
```yaml
privacy_zones:
  - label: "Strada pubblica"
    x1: 0
    y1: 0
    x2: 420
    y2: 180
  - label: "Vicino targa"
    x1: 900
    y1: 400
    x2: 1280
    y2: 720
```

Coordinates are in pixels relative to the camera's native resolution.

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

Add plates in configuration tab and adjust other fields based on your needs.

### 5. Start Addon

First start downloads the AI super-resolution model (~1.5MB) and loads TrOCR. If TrOCR is unavailable, EasyOCR is loaded as fallback. This may take 2-3 minutes on first boot.

### 6. Test (optional)

Enable debug mode and approach the gate. Check logs:
```
Motion detected! Multi-attempt recognition...
  Attempt 1/3: 'GT234YY' (YOLO: 0.821, OCR: 0.954)
  ⚡ Early exit: exact match at attempt 1/3
📊 Best detection: 'GT234YY' (YOLO: 0.821, OCR: 0.954) [early exit]
✅ Exact match 'GT234YY' → gate opening
```

## Troubleshooting

### No plates detected

1. Enable debug mode
2. Check debug image: `/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg`
3. Adjust ROI to focus on plate area
4. Lower confidence threshold (try 0.25)

### Wrong plate detected

1. Check lighting — add illumination if too dark
2. Verify camera angle — plate should be clearly visible
3. Check debug logs for OCR results per attempt
4. The system selects the best result across 3 attempts by combined YOLO + OCR score

### Gate opens for wrong vehicles

1. Verify allowed_plates list is correct
2. Check if plates are similar (e.g., `GT234YY` vs `CT234YY`)
3. Increase confidence threshold
4. Adjust ROI to exclude other vehicles

### Visitor notification not received

1. Verify `notify_services` matches exactly the service name in HA (check Developer Tools → Services)
2. Ensure the Companion App is installed and notifications are enabled on the device
3. Check that `vehicle_detected` was triggered — notifications only fire if a plate was detected during the motion-on phase

## Performance

- **CPU**: 2-3 seconds per attempt (up to 3 attempts, early exit on exact match)
- **GPU**: Not required (CPU-only operation)
- **Memory**: ~200MB with TrOCR · ~700MB with EasyOCR fallback
- **Storage**: ~200MB for models

## Privacy

All processing is done locally on your Home Assistant instance. No data is sent to external services. Privacy zones allow you to mask sensitive areas before any image is stored or sent as a notification.

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
