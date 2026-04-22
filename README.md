# Vision-Based Truck Load Weight Estimation

A real-time truck load weight estimation system deployed on **BeagleY-AI (TI AM67A)** edge AI board.

## What it does
- Captures top and side view images of a loaded truck using dual USB cameras
- Detects the truck bed and cargo using custom YOLOv8 ONNX models
- Estimates load volume using DepthAnything V2 depth estimation
- Measures truck bed height using ultrasonic sensors
- Calculates load weight using material density lookup
- Streams live results via HTTP on the local network

## Achieved Results
- Under 2% estimation error against ground truth weight

## Hardware
- BeagleY-AI (TI AM67A SoC)
- 2x USB cameras (top view + side view)
- 2x Ultrasonic sensors (HC-SR04) via GPIO

## Models Used
- YOLOv8 (custom trained) — truck bed detection, loading area, material classification
- DepthAnything V2 — monocular depth estimation
- All models converted to ONNX and optimized for edge deployment

## Tech Stack
- Python, OpenCV, ONNX Runtime
- gpiod for GPIO/ultrasonic control
- HTTP server for live result streaming

## How to Run
```bash
python truck_pipeline_beagley.py
```
Then open browser at `http://<BeagleY-AI-IP>:8080`

To find your board's IP:
```bash
ip addr show
```

> **Note:** Requires BeagleY-AI hardware with models placed in `models/` folder.

## Calibration
Before each session, calibration must be done according to your physical setup:

In `truck_pipeline_beagley.py`, update these values in the `CONFIG` section:

| Parameter | Description | Default |
|---|---|---|
| `TOP_VIEW_PIXEL_AREA_CM2` | Area per pixel from top camera (measure at your mount height) | 0.000751 |
| `TOP_VIEW_CAMERA_HEIGHT_M` | Height of top camera above ground (in metres) | 0.32 |
| `SIDE_VIEW_PIXEL_AREA_CM2` | Area per pixel from side camera (measure at your mount distance) | 0.000343 |
| `FALLBACK_BED_HEIGHT_M` | Truck bed height above ground if ultrasonic fails | 0.035 |
| `FALLBACK_TOP_BASELINE_M` | Camera height minus bed height if ultrasonic fails | 0.285 |

These values were calibrated for our specific test setup — **you must re-measure and update them for your hardware.**

## Dependencies
```bash
pip install opencv-python onnxruntime numpy gpiod
```