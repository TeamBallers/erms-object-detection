# YOLO Server (FastAPI)

This folder contains the backend server that runs YOLO object detection.

---

## What it does

- Receives images from Raspberry Pi (client)
- Runs YOLO inference
- Saves:
  - raw images
  - annotated images
- Returns detections via JSON

---

# 🚀 Setup Instructions

## 1. Install dependencies

Run the following command inside the `server/` directory:

```bash
pip install -r requirements.txt


