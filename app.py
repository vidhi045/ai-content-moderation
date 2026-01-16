from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
from PIL import Image
import io
import tempfile
import os
import cv2
import json
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from bson import ObjectId
from datetime import datetime
import time

app = FastAPI(title="AI Image & Video Moderation API")

MONGO_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URI)
db = client["ai_moderation"]

# Async GridFS bucket
fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")

# Metadata collection
results_collection = db["image_results"]

# ---------------- CONFIGURATION ----------------

UNSAFE_THRESHOLD = 0.35
REVIEW_THRESHOLD = 0.25
VIDEO_BLOCK_PERCENT = 30
FPS_INTERVAL = 1

MODEL_NAME = "openai/clip-vit-large-patch14"

# ---------------- LOAD MODEL ----------------

pipe = pipeline(
    "zero-shot-image-classification",
    model=MODEL_NAME,
    framework="pt"
)

# ---------------- LOAD LABELS ----------------

def load_labels():
    try:
        with open("labels.json", "r") as f:
            data = json.load(f)

        return (
            data["WEAPON_LABELS"],
            data["UNSAFE_LABELS"],
            data["SAFE_LABELS"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load labels.json: {str(e)}")


WEAPON_LABELS, UNSAFE_LABELS, SAFE_LABELS = load_labels()

# IMPORTANT FIX: include ALL categories
ALL_LABELS = WEAPON_LABELS + UNSAFE_LABELS + SAFE_LABELS

# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "service": "AI Image & Video Moderation",
        "video_enabled": True
    }


# ---------------- UTIL FUNCTIONS ----------------

def read_image(file_bytes):
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def extract_frames(video_path, fps_interval=FPS_INTERVAL):
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise HTTPException(status_code=400, detail="Invalid or corrupted video file")

    frames = []
    fps = vid.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * fps_interval), 1)

    count = 0
    success, frame = vid.read()

    while success:
        if count % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        success, frame = vid.read()
        count += 1

    vid.release()
    return frames


def classify_image_full(image: Image.Image):
    results = pipe(image, candidate_labels=ALL_LABELS)

    unsafe_scores = []
    safe_scores = []
    critical_scores = []

    for r in results:
        label = r["label"]
        score = r["score"]

        if label in WEAPON_LABELS:
            critical_scores.append((label, score))

        elif label in UNSAFE_LABELS:
            unsafe_scores.append((label, score))

        elif label in SAFE_LABELS:
            safe_scores.append((label, score))

    return {
        "all_results": results,
        "max_unsafe_score": max([s for _, s in unsafe_scores], default=0),
        "max_critical_score": max([s for _, s in critical_scores], default=0),
        "total_safe_score": sum([s for _, s in safe_scores]),
        "unsafe_labels": [l for l, _ in unsafe_scores],
        "safe_labels": [l for l, _ in safe_scores],
        "top_predictions": sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    }

# ---------------- IMAGE MODERATION ----------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    start_time = time.time()

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # ----- STORE IMAGE IN GRIDFS ASYNC -----
    try:
        image_stream = io.BytesIO(contents)
        image_id = await fs_bucket.upload_from_stream(file.filename, image_stream)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store image in GridFS: {str(e)}")

    # ----- RUN IMAGE CLASSIFICATION -----
    image = read_image(contents)

    analysis = classify_image_full(image)

    max_critical = analysis["max_critical_score"]
    max_unsafe = analysis["max_unsafe_score"]
    total_safe = analysis["total_safe_score"]

    # ----- DECISION LOGIC -----

    if max_critical >= 0.20:
        decision = "BLOCK"
        reason = "Critical content detected"
        category = "CRITICAL"

    elif max_unsafe >= UNSAFE_THRESHOLD:
        decision = "BLOCK"
        reason = "Unsafe content above confidence threshold"
        category = "UNSAFE"

    else:
        decision = "SAFE"
        reason = "No unsafe content above threshold"
        category = "SAFE"

    end_time = time.time()
    processing_time = round(end_time - start_time, 3)

    # ----- PREPARE METADATA -----
    metadata = {
        "image_id": image_id,
        "filename": file.filename,
        "decision": decision,
        "category": category,
        "processing_time_seconds": processing_time,
        "max_critical_confidence": round(max_critical, 3),
        "max_unsafe_confidence": round(max_unsafe, 3),
        "total_safe_confidence": round(total_safe, 3),
        "unsafe_labels": list(set(analysis["unsafe_labels"])),
        "safe_labels": list(set(analysis["safe_labels"])),
        "top_predictions": analysis["top_predictions"],
        "created_at": datetime.utcnow()
    }

    await results_collection.insert_one(metadata)

    return {
        "image_id": str(image_id),
        "filename": file.filename,
        "decision": decision,
        "reason": reason,
        "category": category,
        "processing_time_seconds": processing_time,
        "max_critical_confidence": round(max_critical, 3),
        "max_unsafe_confidence": round(max_unsafe, 3),
        "total_safe_confidence": round(total_safe, 3),
        "unsafe_labels_detected": list(set(analysis["unsafe_labels"])),
        "safe_labels_detected": list(set(analysis["safe_labels"])),
        "top_predictions": analysis["top_predictions"]
    }

# ---------------- VIDEO MODERATION ----------------
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    start_time = time.time()

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files allowed")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        frames = extract_frames(video_path)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    if not frames:
        raise HTTPException(status_code=400, detail="No readable frames found in video")

    unsafe_frames = 0
    safe_frames = 0

    unsafe_labels = set()
    safe_labels = set()

    for frame in frames:

        analysis = classify_image_full(frame)

        max_critical = analysis["max_critical_score"]
        max_unsafe = analysis["max_unsafe_score"]

        if max_critical >= 0.20:
            unsafe_frames += 1
            unsafe_labels.update(analysis["unsafe_labels"])
            continue

        if max_unsafe >= UNSAFE_THRESHOLD:
            unsafe_frames += 1
            unsafe_labels.update(analysis["unsafe_labels"])
        else:
            safe_frames += 1
            safe_labels.update(analysis["safe_labels"])

    total = len(frames)
    unsafe_percent = (unsafe_frames / total) * 100 if total else 0

    decision = "BLOCK" if unsafe_percent >= VIDEO_BLOCK_PERCENT else "SAFE"

    end_time = time.time()
    processing_time = round(end_time - start_time, 3)

    return {
        "filename": file.filename,
        "frames_analyzed": total,
        "unsafe_frames": unsafe_frames,
        "safe_frames": safe_frames,
        "unsafe_percentage": round(unsafe_percent, 2),
        "decision": decision,
        "processing_time_seconds": processing_time,
        "unsafe_labels_detected": list(unsafe_labels),
        "safe_labels_detected": list(safe_labels)
    }
