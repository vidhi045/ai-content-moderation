from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
from PIL import Image
import io
import tempfile
import os
import cv2

app = FastAPI(title="AI Image & Video Moderation API")

# ---------------- LOAD MODEL ----------------
pipe = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-large-patch14",
    framework="pt"
)

# ---------------- RISK CATEGORIES ----------------

# Anything here = immediate block
HARD_BLOCK = [
    "weapon",
    "gun or firearm",
    "knife or sharp object",
    "violence or physical assault",
    "blood or injury",
    "gore or dead body",
    "fight",
    "crime",
    "abuse",
    "self harm",
    "suicide",
    "terrorist",
]

# Allowed unless dominant
SOFT_BLOCK = [
    "nudity or sexual content",
    "explicit adult content",
    "drug or illegal activity",
    "hate or extremist symbol"
]

# Normal safe context
SAFE_LABELS = [
    # Lighting / camera
    "night time scene", "dark environment", "low light photo", "blurry photo",
    "cctv footage", "security camera", "webcam image", "surveillance video",

    # People / social
    "people standing", "people sitting", "walking people",
    "friends", "family", "group photo", "crowd",
    "selfie", "portrait photo", "fully clothed people",

    # Places
    "indoor room", "street", "road", "building", "house",
    "park", "beach", "restaurant", "kitchen", "office", "gym",

    # Nature
    "trees", "sky", "nature", "outdoor scene",

    # Devices / screens
    "phone", "laptop", "computer screen", "text", "screenshots",

    # Food / cooking (CRITICAL for knife safety)
    "food", "cooking", "food preparation", "meal preparation",
    "chef", "home cooking", "professional chef cooking",
    "cutting vegetables", "cutting fruit", "chopping food",
    "slicing food", "kitchen cooking", "recipe video",

    # Furniture / objects
    "table", "chair", "household objects",

    # Animals
    "animals", "pet",

    # Sports / physical activity (prevents fight false-positives)
    "sports match", "boxing match", "martial arts training",
    "wrestling match", "karate practice", "taekwondo training",
    "gym workout", "fitness training", "dance battle", "stage performance",

    # Medical / hospital
    "medical procedure", "doctor treating patient", "hospital scene",
    "surgery room", "first aid", "wound cleaning",
    "vaccination", "medical training",

    # News / reporting
    "news report", "tv broadcast", "journalist reporting",
    "live news", "breaking news"
]


ALL_LABELS = HARD_BLOCK + SOFT_BLOCK + SAFE_LABELS
THRESHOLD = 0.30

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "CLIP ViT-Large",
        "service": "AI Image & Video Moderation",
        "video_enabled": True,
        "rules": "Hard-risk always blocks"
    }

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, fps_interval=1):
    vid = cv2.VideoCapture(video_path)
    frames = []
    fps = vid.get(cv2.CAP_PROP_FPS)
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

# ---------------- IMAGE MODERATION ----------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = pipe(image, candidate_labels=ALL_LABELS)

    top = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    hard = []
    soft = []

    for r in top:
        if r["score"] >= THRESHOLD:
            if r["label"] in HARD_BLOCK:
                hard.append(r["label"])
            if r["label"] in SOFT_BLOCK:
                soft.append(r["label"])

    decision = "BLOCK" if hard else "SAFE"

    return {
        "filename": file.filename,
        "decision": decision,
        "hard_risk": list(set(hard)),
        "soft_risk": list(set(soft)),
        "top_predictions": top
    }

# ---------------- VIDEO MODERATION ----------------
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    frames = extract_frames(video_path)
    os.remove(video_path)

    if not frames:
        return {
            "filename": file.filename,
            "decision": "SAFE",
            "safe_reason": "No frames could be extracted",
            "frames_analyzed": 0
        }

    hard_detected = set()
    soft_detected = set()
    soft_frames = 0

    for frame in frames:
        results = pipe(frame, candidate_labels=ALL_LABELS)
        top5 = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

        frame_soft = False

        for r in top5:
            if r["score"] >= THRESHOLD:
                if r["label"] in HARD_BLOCK:
                    hard_detected.add(r["label"])
                if r["label"] in SOFT_BLOCK:
                    soft_detected.add(r["label"])
                    frame_soft = True

        if frame_soft:
            soft_frames += 1

    flagged_percentage = (soft_frames / len(frames)) * 100

    # 🚨 Decision logic
    if hard_detected:
        decision = "BLOCK"
        block_reason = "Hard risk detected: " + ", ".join(hard_detected)
        safe_reason = None

    elif flagged_percentage >= 50:
        decision = "BLOCK"
        block_reason = f"Soft risk in {round(flagged_percentage,2)}% of frames"
        safe_reason = None

    else:
        decision = "SAFE"
        block_reason = None
        safe_reason = "No violence, weapons, or dominant unsafe content detected"

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "frames_analyzed": len(frames),
        "flagged_percentage": round(flagged_percentage, 2),
        "decision": decision,
        "block_reason": block_reason,
        "safe_reason": safe_reason,
        "hard_risk_detected": list(hard_detected),
        "soft_risk_detected": list(soft_detected)
    }
