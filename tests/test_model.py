import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Make sure a model file exists before app.py is imported."""
    if not os.path.exists(train.MODEL_PATH):
        train.train()


@pytest.fixture()
def client():
    import app as app_module

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_model_accuracy(tmp_path):
    accuracy = train.train(str(tmp_path / "m.joblib"))
    assert accuracy > 0.9


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_setosa(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert response.get_json()["species"] == "setosa"


def test_predict_bad_input(client):
    assert client.post("/predict", json={}).status_code == 400
    assert client.post("/predict", json={"features": [1, 2]}).status_code == 400
    assert client.post("/predict", json={"features": ["a", "b", "c", "d"]}).status_code == 400
