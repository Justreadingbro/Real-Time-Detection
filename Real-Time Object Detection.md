## Real-Time Object Detection with YOLOv8 + Flask

> A real-time object detection web app using **YOLOv8**, **OpenCV**, and **Flask**.  
> Captures live video (webcam or external camera), runs YOLO inference, and streams annotated frames over a LAN-accessible web page.

---

## 🚀 Features
- Real-time object detection using **YOLOv8** (default: `yolov8n.pt`)  
- Streams annotated frames via a simple **Flask MJPEG** endpoint (works on desktop and mobile browsers on the same network)  
- FPS overlay and optional video saving  
- Threaded capture + inference pipeline for smoother live responsiveness  
- Easy to swap models (`yolov8n`, `yolov8s`, `yolov8m`, etc.)

---

## 📋 Prerequisites
- Python 3.9+  
- (Recommended) GPU + CUDA for high FPS — install appropriate `torch` wheel for your CUDA version.  
- `ffmpeg` is optional only if you want advanced video handling (not required for basic usage).

---

# 🛠️ Installation

## Clone the repo:
```bash
git clone https://github.com/Justreadingbro/Real-Time-Detection.git
cd realtime-detection
```
## Create & activate virtual environment (example using venv or conda):

```venv
python -m venv rl_env
```
### Windows
```bash
rl_env\Scripts\activate
```
### macOS / Linux
```bash
source rl_env/bin/activate
```
### conda
```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env
```
# Install dependencies:

```bash
pip install -r requirements.txt
```
## **Important: Install torch separately using the official instructions for your platform/[CUDA version](https://pytorch.org/get-started/locally/)**
## Example (CPU-only):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
## ▶️ Usage
### Start the app (default webcam)
```bash
python app.py --source 0 --model yolov8n.pt --port 5000
```
# Common examples
### Default laptop webcam:

```bash
python app.py --source 0
```
### External / virtual camera (e.g., Phone Link / DroidCam):

```bash
python app.py --source 1
```
#### If 1 doesn’t match, try 2, 3, etc. See Troubleshooting below to detect available indices.

## Video file:

```bash
python app.py --source sample.mp4
```
#### Save annotated output:

```bash
python app.py --source 0 --save output/result.mp4
```
#### Set capture resolution:

```bash
python app.py --source 0 --width 640 --height 480
```
## CLI options:

--source : 0 (webcam), integer index, or path to video file

--model : YOLOv8 model path or name (default: yolov8n.pt)

--port : Flask port (default 5000)

--width : Optional capture width (e.g., 640)

--height : Optional capture height (e.g., 480)

--save : Path to save annotated output video

## 📱 Access from Mobile (LAN):

Make sure your PC and mobile are on the same network.

Get your PC's local IP (example Windows ipconfig → IPv4 Address), e.g. 192.168.29.102.

## Open on mobile browser:

```bash
http://<PC_IP>:5000
```

You should see the web page and live annotated feed.

If you cannot connect from mobile: check your PC firewall and allow inbound connections on the chosen port (default 5000).

## 🔍 Troubleshooting
Camera index confusion
Some virtual cameras (Phone Link, DroidCam, OBS VirtualCam) appear at different indices. Run this script to check available camera indices:

```python
import cv2
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        cap.release()
```
# Blank or frozen frames on phone

Ensure Flask is running and the mobile device is on the same Wi-Fi.

Lower resolution (--width 640 --height 480) or use yolov8n for higher FPS.

# Performance

Use GPU-enabled torch for best FPS.

Use yolov8n.pt (nano) for maximum speed; yolov8s/m/l trade speed for accuracy.

Consider exporting the model to ONNX/TensorRT for further optimization.

# Port already in use

Change port: python app.py --port 6000

# 🧾 License
This repository is released under the GNU General Public License v3.0 (GPL-3.0). See the included LICENSE file for the full text.

Note: this project uses YOLOv8 (Ultralytics). Respect the license of third-party components and models you include or distribute.

# 🙏 Acknowledgements

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
[OpenCV](https://opencv.org/)
[Flask](https://flask.palletsprojects.com/en/stable/)