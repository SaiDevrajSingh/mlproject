import os
import sys
import logging
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional
from dotenv import load_dotenv


APP_NAME = "csk-prematch-service"

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(APP_NAME)

# MODEL_PATH can be set as an environment variable (e.g., GCS path downloaded at startup or local path)
# .env support
load_dotenv()

DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "artifacts" / "csk_prematch_random_forest.pkl")
)

app = FastAPI(title=APP_NAME, version="1.0.0")


class PreMatchInput(BaseModel):
    season: int = Field(..., description="Season year, e.g., 2025")
    venue: str
    city: str
    stage: str
    match_number: int
    opponent: str

    @validator('season')
    def _season_range(cls, v):
        if v < 2008 or v > 2100:
            raise ValueError('season must be between 2008 and 2100')
        return v


class PostTossInput(PreMatchInput):
    toss_winner: str
    toss_decision: str


class PredictionResponse(BaseModel):
    win_probability: float
    predicted_win: bool
    model_name: Optional[str] = None
    model_version_created_at: Optional[int] = None


model_obj = None
pipeline = None
model_name = None
model_created_at = None
feature_cols = None
use_post_toss = False


def _load_model(model_path: str):
    global model_obj, pipeline, model_name, model_created_at, feature_cols, use_post_toss
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model_obj = joblib.load(model_path)
    pipeline = model_obj.get("pipeline")
    if pipeline is None:
        raise RuntimeError("Invalid model artifact: missing 'pipeline'")
    model_name = getattr(pipeline, "__class__", type("obj", (), {})).__name__
    model_created_at = int(model_obj.get("created_at", 0))
    feature_cols = model_obj.get("feature_cols")
    use_post_toss = bool(model_obj.get("use_post_toss", False))
    logger.info("Model loaded: %s | created_at=%s | use_post_toss=%s", model_name, model_created_at, use_post_toss)


@app.on_event("startup")
def on_startup():
    logger.info("Starting up %s", APP_NAME)
    _load_model(DEFAULT_MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None, "use_post_toss": use_post_toss}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _check_auth(authorization: Optional[str]):
    required = os.getenv("API_TOKEN")
    if not required:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization.split(" ", 1)[1]
    allowed = {t.strip() for t in os.getenv("ALLOWED_TOKENS", "").split(",") if t.strip()}
    if allowed:
        return token in allowed
    return token == required


@app.middleware("http")
async def timing_and_logging(request: Request, call_next):
    import time

    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("%s %s -> %s in %.2f ms", request.method, request.url.path, request.client.host if request.client else "?", elapsed_ms)


@app.post("/predict-prematch", response_model=PredictionResponse)
def predict_prematch(payload: PreMatchInput, authorization: Optional[str] = Header(None)):
    if not _check_auth(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        import pandas as pd
        X = pd.DataFrame({
            'season': [payload.season],
            'venue': [payload.venue],
            'city': [payload.city],
            'stage': [payload.stage],
            'match_number': [payload.match_number],
            'opponent': [payload.opponent]
        })
        proba = float(pipeline.predict_proba(X)[:, 1][0])
        return PredictionResponse(
            win_probability=proba,
            predicted_win=proba >= 0.5,
            model_name=model_name,
            model_version_created_at=model_created_at,
        )
    except Exception as e:
        logger.warning("Bad request: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-posttoss", response_model=PredictionResponse)
def predict_posttoss(payload: PostTossInput, authorization: Optional[str] = Header(None)):
    if not _check_auth(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not use_post_toss:
        raise HTTPException(status_code=400, detail="Loaded model is pre-match only; retrain with toss features to enable post-toss predictions")
    # Ensure model expects toss fields
    if feature_cols and ("toss_winner" not in feature_cols or "toss_decision" not in feature_cols):
        raise HTTPException(status_code=400, detail="Model artifact missing toss features")
    try:
        import pandas as pd
        X = pd.DataFrame({
            'season': [payload.season],
            'venue': [payload.venue],
            'city': [payload.city],
            'stage': [payload.stage],
            'match_number': [payload.match_number],
            'opponent': [payload.opponent],
            'toss_winner': [payload.toss_winner],
            'toss_decision': [payload.toss_decision]
        })
        proba = float(pipeline.predict_proba(X)[:, 1][0])
        return PredictionResponse(
            win_probability=proba,
            predicted_win=proba >= 0.5,
            model_name=model_name,
            model_version_created_at=model_created_at,
        )
    except Exception as e:
        logger.warning("Bad request: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))


