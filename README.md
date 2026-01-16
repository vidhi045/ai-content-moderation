# AI Content Moderation API

FastAPI-based AI moderation system for:

- Image moderation
- Video moderation
- MongoDB + GridFS storage
- CLIP zero-shot classification

## Setup

1. Create virtual environment

python -m venv .venv
source .venv/bin/activate   (Linux/Mac)
.venv\Scripts\activate      (Windows)

2. Install dependencies

pip install -r requirements.txt

3. Run server

uvicorn app:app --reload
