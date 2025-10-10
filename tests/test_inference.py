import os
import pathlib
import joblib
import pandas as pd


def test_model_artifact_loads():
    root = pathlib.Path(__file__).resolve().parents[1]
    model_path = os.path.join(root, 'artifacts', 'csk_prematch_random_forest.pkl')
    assert os.path.exists(model_path), f"Missing model artifact at {model_path}"
    obj = joblib.load(model_path)
    assert 'pipeline' in obj


def test_inference_shapes():
    root = pathlib.Path(__file__).resolve().parents[1]
    model_path = os.path.join(root, 'artifacts', 'csk_prematch_random_forest.pkl')
    pipe = joblib.load(model_path)['pipeline']
    X = pd.DataFrame({
        'season': [2025],
        'venue': ['MA Chidambaram Stadium, Chepauk'],
        'city': ['Chennai'],
        'stage': ['league'],
        'match_number': [10],
        'opponent': ['Mumbai Indians']
    })
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (1,)

