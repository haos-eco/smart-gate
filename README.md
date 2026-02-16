# Smart Gate - Automatic License Plate Recognition

Automatic gate opener using YOLO + EasyOCR for license plate recognition.

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg


## Installation

### Step 1: Add Repository

1. Go to **Settings → Add-ons → Add-on Store → Three dots → Repositories**
2. Add: `https://github.com/andreaemmanuele/haos_smartgate`

### Step 2: Install Add-on

Find "Smart Gate" in the store and click **Install**

### Step 3: Download YOLO Model

You need to download a YOLOv11 license plate detection model:

**Option A: Pre-trained model (recommended)**
1. Download from: https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/best.onnx
2. Rename to `model.onnx`

**Option B: Train your own**
Train a custom YOLOv11 model on your specific plates/cameras

### Step 4: Upload Model

Upload `model.onnx` to Home Assistant:

**Via File Editor addon:**
1. Create folder: `/config/www/smart_gate/`
2. Upload `model.onnx` there

**Via Samba/SSH:**
```bash
scp model.onnx root@HOMEASSISTANT_IP:/config/www/smart_gate/
```

**Via WebUI:**
1. Install "File Editor" addon
2. Create `/config/www/smart_gate/` folder
3. Upload file

### Step 5: Configure

1. Open Smart Gate addon
2. Configure:
    - **Motion sensor**: Binary sensor that triggers on vehicle detection
    - **Camera entity**: Camera to capture license plates
    - **Gate switch**: Switch entity to open the gate
    - **Allowed plates**: List of authorized license plates (format: AB123CD)
    - **Model path**: `/config/www/smart_gate/model.onnx`

### Step 6: Start

Click **Start** and check the logs!

## Configuration
```yaml
motion_entity: binary_sensor.gate_motion
camera_entity: camera.gate_camera
gate_switch: switch.gate_relay
model_path: /config/www/smart_gate/model.onnx
allowed_plates:
  - AB123CD
confidence: 0.5
```

## Troubleshooting

**Error: "Model not found"**
- Verify the file exists at `/config/www/smart_gate/model.onnx`
- Check file size is ~200MB (not a few KB)

**No plates detected**
- Lower confidence threshold (try 0.3)
- Check camera image quality
- Test with daytime images first
