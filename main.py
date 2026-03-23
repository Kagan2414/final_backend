"""
SafeGuard YOLOv8 Accident Detection Server

FastAPI backend that uses YOLOv8 for real-time accident detection.
Accepts base64-encoded video frames and returns detection results.

Usage:
    pip install -r requirements.txt
    python main.py
    # Server runs at http://localhost:8000
"""

import os
import io
import base64
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
CUSTOM_MODEL_PATH = os.environ.get("YOLO_CUSTOM_MODEL_PATH", "runs/detect/accident_model/weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE", "0.40"))

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_CLASS = "person"
ACCIDENT_KEYWORDS = {"accident", "crash", "collision", "fire", "smoke", "damage"}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model: YOLO | None = None
model_type: str = "none"
model_load_time: float = 0

def load_model():
    """Load the best available YOLO model."""
    global model, model_type, model_load_time

    start = time.time()

    custom_path = Path(CUSTOM_MODEL_PATH)
    if custom_path.exists():
        model = YOLO(str(custom_path))
        model_type = "custom_accident"
        print(f"[SafeGuard] Loaded custom accident model from {custom_path}")
    else:
        model = YOLO(MODEL_PATH)
        model_type = "yolov8_pretrained"
        print(f"[SafeGuard] Loaded pre-trained model: {MODEL_PATH}")

    model_load_time = time.time() - start
    print(f"[SafeGuard] Model loaded in {model_load_time:.2f}s")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="SafeGuard YOLOv8 Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class DetectRequest(BaseModel):
    frame_data: str
    sensitivity: float = 0.5

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]

class DetectResponse(BaseModel):
    is_accident: bool
    confidence: float
    severity: str
    accident_type: str
    detections: list[Detection]
    vehicle_count: int
    person_count: int
    inference_ms: float
    model_type: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    model_path: str
    load_time_s: float
    uptime_s: float

# ---------------------------------------------------------------------------
# Accident analysis logic
# ---------------------------------------------------------------------------
_server_start = time.time()

def classify_severity(confidence: float) -> str:
    if confidence >= 0.80:
        return "severe"
    if confidence >= 0.55:
        return "moderate"
    return "minor"

def determine_accident_type(detections: list[Detection]) -> str:
    class_names = {d.class_name.lower() for d in detections}
    if class_names & {"fire", "smoke"}:
        return "fire"
    vehicles = [d for d in detections if d.class_name.lower() in VEHICLE_CLASSES]
    if len(vehicles) >= 2:
        boxes = [d.bbox for d in vehicles]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _boxes_overlap(boxes[i], boxes[j]):
                    return "collision"
    persons = [d for d in detections if d.class_name.lower() == PERSON_CLASS]
    if persons and vehicles:
        for p in persons:
            for v in vehicles:
                if _boxes_near(p.bbox, v.bbox):
                    return "pedestrian_incident"
    if vehicles:
        return "vehicle_incident"
    return "unknown"

def _boxes_overlap(b1: list[float], b2: list[float], threshold: float = 0.15) -> bool:
    """Check if two bounding boxes overlap beyond a threshold (IoU)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    if union == 0:
        return False
    return (intersection / union) > threshold

def _boxes_near(b1: list[float], b2: list[float], margin: float = 50) -> bool:
    """Check if two bounding boxes are near each other."""
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
    return dist < margin

def analyze_scene(detections: list[Detection], sensitivity: float) -> tuple[bool, float, str, str]:
    """
    Analyze detections to determine if an accident occurred.
    Returns (is_accident, confidence, severity, accident_type).
    """
    if not detections:
        return False, 0.0, "none", "none"

    vehicles = [d for d in detections if d.class_name.lower() in VEHICLE_CLASSES]
    persons = [d for d in detections if d.class_name.lower() == PERSON_CLASS]
    hazards = [d for d in detections if d.class_name.lower() in {"fire", "smoke"}]

    # Custom model: classes may include "accident" directly
    accident_dets = [d for d in detections if d.class_name.lower() in ACCIDENT_KEYWORDS]
    if accident_dets:
        conf = max(d.confidence for d in accident_dets)
        sev = classify_severity(conf)
        atype = determine_accident_type(detections)
        return True, conf, sev, atype

    accident_score = 0.0

    # Multiple vehicles close together
    if len(vehicles) >= 2:
        boxes = [d.bbox for d in vehicles]
        overlaps = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _boxes_overlap(boxes[i], boxes[j]):
                    overlaps += 1
        if overlaps > 0:
            accident_score += 0.4 + (overlaps * 0.15)

    # Person near vehicle
    if persons and vehicles:
        for p in persons:
            for v in vehicles:
                if _boxes_near(p.bbox, v.bbox):
                    accident_score += 0.25
                    break

    # Fire or smoke
    if hazards:
        best_hazard = max(d.confidence for d in hazards)
        accident_score += best_hazard * 0.5

    # High density of objects in small area
    if len(detections) > 5:
        accident_score += 0.1

    adjusted_threshold = max(0.1, 0.5 - (sensitivity - 0.5) * 0.4)
    is_accident = accident_score >= adjusted_threshold
    confidence = min(1.0, accident_score)
    severity = classify_severity(confidence) if is_accident else "none"
    atype = determine_accident_type(detections) if is_accident else "none"

    return is_accident, confidence, severity, atype

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok" if model is not None else "no_model",
        model_loaded=model is not None,
        model_type=model_type,
        model_path=CUSTOM_MODEL_PATH if model_type == "custom_accident" else MODEL_PATH,
        load_time_s=round(model_load_time, 2),
        uptime_s=round(time.time() - _server_start, 1),
    )

@app.post("/detect", response_model=DetectResponse)
async def detect_accident(req: DetectRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        img_data = req.frame_data
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image data: {e}")

    start = time.time()
    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    inference_ms = (time.time() - start) * 1000

    detections: list[Detection] = []
    if results and len(results) > 0:
        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name=names.get(cls_id, f"class_{cls_id}"),
                confidence=round(conf, 4),
                bbox=[round(v, 1) for v in xyxy],
            ))

    vehicle_count = sum(1 for d in detections if d.class_name.lower() in VEHICLE_CLASSES)
    person_count = sum(1 for d in detections if d.class_name.lower() == PERSON_CLASS)

    is_accident, confidence, severity, accident_type = analyze_scene(
        detections, req.sensitivity
    )

    return DetectResponse(
        is_accident=is_accident,
        confidence=round(confidence, 4),
        severity=severity,
        accident_type=accident_type,
        detections=detections,
        vehicle_count=vehicle_count,
        person_count=person_count,
        inference_ms=round(inference_ms, 1),
        model_type=model_type,
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f"[SafeGuard] Starting YOLOv8 server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
