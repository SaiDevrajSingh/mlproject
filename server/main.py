import os
import sys
import logging
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse, HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CSK IPL Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #3498db; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2980b9; }
            .result { margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 5px; }
            .win { background: #d5f4e6; border-left: 4px solid #27ae60; }
            .loss { background: #fadbd8; border-left: 4px solid #e74c3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏏 CSK IPL Win Prediction</h1>
            <p>Predict Chennai Super Kings' chances of winning an IPL match!</p>
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <strong>⚠️ Note:</strong> This model was trained on historical data up to 2015. Predictions for seasons after 2015 may be less accurate.
            </div>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Season:</label>
                    <input type="number" id="season" value="2015" min="2008" max="2015">
                </div>
                
                <div class="form-group">
                    <label>Venue:</label>
                    <input type="text" id="venue" value="MA Chidambaram Stadium, Chepauk" placeholder="Enter venue name">
                </div>
                
                <div class="form-group">
                    <label>City:</label>
                    <input type="text" id="city" value="Chennai" placeholder="Enter city">
                </div>
                
                <div class="form-group">
                    <label>Stage:</label>
                    <select id="stage">
                        <option value="league">League</option>
                        <option value="qualifier">Qualifier</option>
                        <option value="eliminator">Eliminator</option>
                        <option value="final">Final</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Match Number:</label>
                    <input type="number" id="match_number" value="10" min="1" max="100">
                </div>
                
                <div class="form-group">
                    <label>Opponent:</label>
                    <input type="text" id="opponent" value="Mumbai Indians" placeholder="Enter opponent team">
                </div>
                
                <button type="submit">🔮 Predict CSK Win Probability</button>
            </form>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const data = {
                    season: parseInt(document.getElementById('season').value),
                    venue: document.getElementById('venue').value,
                    city: document.getElementById('city').value,
                    stage: document.getElementById('stage').value,
                    match_number: parseInt(document.getElementById('match_number').value),
                    opponent: document.getElementById('opponent').value
                };
                
                try {
                    const response = await fetch('/predict-prematch', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (result.predicted_win) {
                        resultDiv.className = 'result win';
                        resultDiv.innerHTML = `
                            <h3>🎉 CSK is predicted to WIN!</h3>
                            <p><strong>Win Probability:</strong> ${(result.win_probability * 100).toFixed(1)}%</p>
                            <p><strong>Confidence:</strong> ${result.win_probability > 0.7 ? 'High' : result.win_probability > 0.5 ? 'Medium' : 'Low'}</p>
                        `;
                    } else {
                        resultDiv.className = 'result loss';
                        resultDiv.innerHTML = `
                            <h3>😔 CSK is predicted to LOSE</h3>
                            <p><strong>Win Probability:</strong> ${(result.win_probability * 100).toFixed(1)}%</p>
                            <p><strong>Confidence:</strong> ${result.win_probability < 0.3 ? 'High' : result.win_probability < 0.5 ? 'Medium' : 'Low'}</p>
                        `;
                    }
                    
                    resultDiv.style.display = 'block';
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """


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
        import numpy as np
        
        # Check if season is within training range
        if payload.season > 2015:  # Model trained on data up to 2009
            logger.warning(f"Season {payload.season} is outside training range (up to 2015)")
            # Return a default prediction with warning
            return PredictionResponse(
                win_probability=0.5,  # Neutral prediction
                predicted_win=False,
                model_name=model_name,
                model_version_created_at=model_created_at,
            )
        
        X = pd.DataFrame({
            'season': [payload.season],
            'venue': [payload.venue],
            'city': [payload.city],
            'stage': [payload.stage],
            'match_number': [payload.match_number],
            'opponent': [payload.opponent]
        })
        
        proba = pipeline.predict_proba(X)[:, 1][0]
        
        # Check for NaN or invalid predictions
        if np.isnan(proba) or np.isinf(proba):
            logger.warning("Model returned NaN/Inf probability")
            proba = 0.5  # Default neutral prediction
            
        return PredictionResponse(
            win_probability=float(proba),
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


