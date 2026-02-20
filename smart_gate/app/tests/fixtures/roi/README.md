# Test ROI Snapshot

Visual test that crops a real snapshot using the production ROI and saves the output for inspection.

## Setup

Add a real snapshot from your gate camera to the fixtures folder:

```
smart_gate/app/tests/fixtures/roi/latest.jpg
```

You can grab one directly from Home Assistant:

```
http://<ha-ip>:8123/local/smart_gate/snapshot/latest.jpg
```

## Configuration

Edit the constants at the top of `test_roi_snapshot.py` to match your setup:

```python
SNAPSHOT_PATH = "smart_gate/app/tests/fixtures/roi/latest.jpg"
ROI = [0.25, 0.15, 0.55, 0.50]  # x, y, w, h in relative floats (0.0 to 1.0)
OUTPUT_PATH = "smart_gate/app/tests/fixtures/roi/latest_roi_output.jpg"
```

## Run

From the project root:

```bash
pytest -s smart_gate/app/tests/test_roi_snapshot.py
```

The `-s` flag prints the original and cropped dimensions to the console:

```
Original size: 1920x1080
Cropped size:  1056x540
ROI applied:   x=0.25, y=0.15, w=0.55, h=0.50
Output saved:  smart_gate/app/tests/fixtures/roi/latest_roi_output.jpg
```

## Output

The cropped image is saved to `smart_gate/app/tests/fixtures/roi/latest_roi_output.jpg`. Open it to visually verify the ROI is covering the correct area of the frame (the gate entrance, excluding sky, vegetation and gate post).

If the crop looks wrong, adjust the ROI values and re-run until the framing is correct, then update `config.yaml` with the same values:

```yaml
roi: "0.25,0.15,0.55,0.50"
```
