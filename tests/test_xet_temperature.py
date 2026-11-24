import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xet_temperature import predict, train  # noqa: E402

def test_model_learns(tmp_path):
    mae, r2 = train.train(str(tmp_path / "xet.joblib"))
    assert r2 > 0.9
    assert mae < 3

def test_hotter_when_busier(tmp_path):
    path = str(tmp_path / "xet.joblib")
    train.train(path)
    idle = {"cpu_percent": 5, "mem_percent": 30, "load_1min": 0.2}
    busy = {"cpu_percent": 95, "mem_percent": 80, "load_1min": 7.0}
    assert predict.estimate(busy, path) > predict.estimate(idle, path) + 20
