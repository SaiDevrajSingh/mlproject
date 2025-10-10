# CSK IPL Performance Prediction - Production Serving

## FastAPI Service (Cloud Run)

- Build model artifacts in the notebook (Step F/H):
  - `artifacts/csk_prematch_<model>.pkl`
  - `artifacts/model_card.json`

- Windows .venv setup (PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r server\requirements.txt
```

- Env vars (.env supported)
  - Create `server/.env` or project `.env` with:
```
MODEL_PATH=artifacts/csk_prematch_random_forest.pkl
LOG_LEVEL=INFO
```

- Local run
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8080
```

- Docker
```bash
docker build -t csk-prematch:latest -f server/Dockerfile .
docker run -p 8080:8080 csk-prematch:latest
```

- API
  - GET `/health`
  - POST `/predict-prematch`
  - POST `/predict-posttoss` (requires toss-aware model)
```json
{
  "season": 2025,
  "venue": "MA Chidambaram Stadium, Chepauk",
  "city": "Chennai",
  "stage": "league",
  "match_number": 10,
  "opponent": "Mumbai Indians"
}
```

## Firebase Integration

- Frontend: Firebase Hosting (or any web client) → call Cloud Run URL.
- Auth: Protect Cloud Run with Firebase Authentication; pass ID token in `Authorization: Bearer <token>`.
- Firestore (optional): store predictions in `matches` collection with model metadata.

## Deployment to Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/csk-prematch:latest

gcloud run deploy csk-prematch \
  --image gcr.io/PROJECT_ID/csk-prematch:latest \
  --region REGION \
  --platform managed \
  --allow-unauthenticated=false \
  --set-env-vars MODEL_PATH=/app/artifacts/csk_prematch_random_forest.pkl
```

## Notes
- Pre-match model avoids target leakage. For post-toss, retrain with toss fields.
- Use temporal splits for evaluation; monitor input drift and recalibrate as needed.


