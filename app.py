from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io
import tempfile
import os
import cv2
import json
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from datetime import datetime
import time
from dotenv import load_dotenv

# --------------------------------------------------
# APP INITIALIZATION
# --------------------------------------------------

app = FastAPI(title="AI Image & Video Moderation API")

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not found in environment variables")

try:
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["ai-moderation"]

    fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")
    results_collection = db["image_results"]

    video_fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="videos")
    video_results_collection = db["video_results"]

except Exception as e:
    raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}")

# --------------------------------------------------
# GLOBAL ERROR HANDLERS
# --------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"Internal server error: {str(exc)}",
            "data": {}
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": {}
        }
    )

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

UNSAFE_THRESHOLD = 0.35
VIDEO_BLOCK_PERCENT = 30
FPS_INTERVAL = 1

MODEL_NAME = "openai/clip-vit-large-patch14"

# --------------------------------------------------
# MODEL LOADING WITH ERROR HANDLING
# --------------------------------------------------

try:
    pipe = pipeline(
        "zero-shot-image-classification",
        model=MODEL_NAME,
        framework="pt"
    )
except Exception as e:
    raise RuntimeError(f"Failed to load ML model: {str(e)}")

# --------------------------------------------------
# LABEL LOADING
# --------------------------------------------------

def load_labels():
    try:
        with open("labels.json", "r") as f:
            data = json.load(f)

        return (
            data["WEAPON_LABELS"],
            data["UNSAFE_LABELS"],
            data["SAFE_LABELS"]
        )
    except FileNotFoundError:
        raise RuntimeError("labels.json file not found")
    except KeyError as e:
        raise RuntimeError(f"Missing key in labels.json: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load labels.json: {str(e)}")

try:
    WEAPON_LABELS, UNSAFE_LABELS, SAFE_LABELS = load_labels()
    ALL_LABELS = WEAPON_LABELS + UNSAFE_LABELS + SAFE_LABELS
except Exception as e:
    raise RuntimeError(str(e))

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/v1/health")
async def health():
    return {
        "success": True,
        "message": "Service is healthy",
        "data": {
            "model": MODEL_NAME,
            "service": "AI Image & Video Moderation",
            "video_enabled": True
        }
    }

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def read_image(file_bytes):
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

def extract_frames(video_path, fps_interval=FPS_INTERVAL):
    try:
        vid = cv2.VideoCapture(video_path)

        if not vid.isOpened():
            raise Exception("Unable to open video file")

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

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame extraction failed: {str(e)}")

def classify_image_full(image: Image.Image):
    try:
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
            "top_predictions": sorted(results, key=lambda x: x["score"], reverse=True)[:3]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# --------------------------------------------------
# IMAGE CLASSIFICATION
# --------------------------------------------------

@app.post("/api/v1/image_classify")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        try:
            image_stream = io.BytesIO(contents)
            image_id = await fs_bucket.upload_from_stream(file.filename, image_stream)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB storage failed: {str(e)}")

        image = read_image(contents)
        analysis = classify_image_full(image)

        max_critical = analysis["max_critical_score"]
        max_unsafe = analysis["max_unsafe_score"]
        total_safe = analysis["total_safe_score"]

        if max_critical >= 0.20:
            decision = "BLOCK"
            reason = "Critical content detected"
            category = "CRITICAL"
        elif max_unsafe >= UNSAFE_THRESHOLD:
            decision = "BLOCK"
            reason = "Unsafe content above threshold"
            category = "UNSAFE"
        else:
            decision = "SAFE"
            reason = "No unsafe content detected"
            category = "SAFE"

        processing_time = round(time.time() - start_time, 3)

        metadata = {
            "image_id": image_id,
            "filename": file.filename,
            "decision": decision,
            "category": category,
            "processing_time_seconds": processing_time,
            "created_at": datetime.utcnow()
        }

        await results_collection.insert_one(metadata)

        return {
            "success": True,
            "message": "Image classified successfully",
            "data": {
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
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        return {
            "success": False,
            "message": f"Image processing failed: {str(e)}",
            "data": {}
        }

# --------------------------------------------------
# VIDEO CLASSIFICATION
# --------------------------------------------------

@app.post("/api/v1/video_classify")
async def video_classify(file: UploadFile = File(...)):
    start_time = time.time()
    video_path = None

    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Only video files allowed")

        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Store video in GridFS
        video_id = await video_fs_bucket.upload_from_stream(
            file.filename, io.BytesIO(contents)
        )

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            video_path = tmp.name

        # ----- CALCULATE VIDEO DURATION -----
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid or corrupted video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        video_duration = round(frame_count / fps, 2) if fps else 0

        cap.release()
        
        # Extract frames for analysis

        frames = extract_frames(video_path)

        if not frames:
            raise HTTPException(status_code=400, detail="No readable frames found")

        unsafe_frames = 0
        safe_frames = 0

        unsafe_labels = set()
        safe_labels = set()

        total = len(frames)

        for frame in frames:
            analysis = classify_image_full(frame)

            if analysis["max_unsafe_score"] >= UNSAFE_THRESHOLD:
                unsafe_frames += 1
                unsafe_labels.update(analysis["unsafe_labels"])
            else:
                safe_frames += 1
                safe_labels.update(analysis["safe_labels"])

        unsafe_percent = (unsafe_frames / total) * 100 if total else 0

        decision = "BLOCK" if unsafe_percent >= VIDEO_BLOCK_PERCENT else "SAFE"

        processing_time = round(time.time() - start_time, 3)

        # Store result metadata

        await video_results_collection.insert_one({
            "video_id": video_id,
            "filename": file.filename,
            "decision": decision,
            "video_duration_seconds": video_duration,
            "processing_time_seconds": processing_time,
            "frames_analyzed": total,
            "unsafe_frames": unsafe_frames,
            "safe_frames": safe_frames,
            "unsafe_percentage": unsafe_percent,
            "unsafe_labels": list(unsafe_labels),
            "safe_labels": list(safe_labels),
            "created_at": datetime.utcnow()
        })

        return {
            "success": True,
            "message": "Video classified successfully",
            "data": {
                "video_id": str(video_id),
                "filename": file.filename,
                "video_duration_seconds": video_duration,
                "processing_time_seconds": processing_time,
                "frames_analyzed": total,
                "unsafe_frames": unsafe_frames,
                "safe_frames": safe_frames,
                "unsafe_percentage": round(unsafe_percent, 2),
                "decision": decision,
                "unsafe_labels_detected": list(unsafe_labels),
                "safe_labels_detected": list(safe_labels)
            }
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        return {
            "success": False,
            "message": f"Video processing failed: {str(e)}",
            "data": {}
        }

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

@app.get("/test-db")
async def test_db():
    try:
        return {
            "success": True,
            "message": "DB connection successful",
            "data": await client.list_database_names()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"DB connection failed: {str(e)}",
            "data": {}
        }