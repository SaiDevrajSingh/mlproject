import os
import pathlib
import pytest
from fastapi.testclient import TestClient

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from server.main import app


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ.setdefault('MODEL_PATH', str(ROOT / 'artifacts' / 'csk_prematch_random_forest.pkl'))
    # optional token
    os.environ.pop('API_TOKEN', None)
    os.environ.pop('ALLOWED_TOKENS', None)


def test_health():
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200
    assert 'model_loaded' in r.json()


def test_predict_prematch():
    client = TestClient(app)
    payload = {
        'season': 2025,
        'venue': 'MA Chidambaram Stadium, Chepauk',
        'city': 'Chennai',
        'stage': 'league',
        'match_number': 10,
        'opponent': 'Mumbai Indians'
    }
    r = client.post('/predict-prematch', json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body['win_probability'] <= 1.0

