from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from datetime import datetime
from pathlib import Path
import torch

app = FastAPI()

# ----------------------------
# DEVICE (mine will use mps since its a mac)
# ----------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# ----------------------------
# MODEL
# ----------------------------
model = YOLO("yolo11s.pt")
model.to(DEVICE)

# ----------------------------
# FOLDERS
# ----------------------------
BASE = Path("runs/api_outputs")
RAW_DIR = BASE / "raw"
ANN_DIR = BASE / "annotated"

RAW_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# STATUS
# ----------------------------
@app.get("/")
def root():
    return {"status": "YOLO server running", "device": DEVICE}


# ----------------------------
# DETECT
# ----------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # ----------------------------
    # LOAD IMAGE (TRUE RGB)
    # ----------------------------
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # so that the annotated image wont have a blue hue and filter applied
    image_np = np.array(image)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # ----------------------------
    # SAVE RAW (exact original)
    # ----------------------------
    raw_path = RAW_DIR / f"raw_{timestamp}.jpg"
    image.save(raw_path)

    # ----------------------------
    # RUN YOLO
    # ----------------------------
    results = model(image_np)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": conf,
            "bbox": box.xyxy[0].tolist()
        })

    # ----------------------------
    # CREATE ANNOTATED IMAGE 
    # ----------------------------

    # this returns rgb image so that there is no weird blue hue and filter
    annotated_img = r.plot()

    # convert numpy to PIL (so that there is no color shift)
    annotated_pil = Image.fromarray(annotated_img)

    annotated_path = ANN_DIR / f"annotated_{timestamp}.jpg"
    annotated_pil.save(annotated_path)

    # ----------------------------
    # JSON RESPONSE
    # ----------------------------
    return {
        "detections": detections,
        "num_detections": len(detections),
        "raw_image": str(raw_path),
        "annotated_image": str(annotated_path),
        "device": DEVICE
    }