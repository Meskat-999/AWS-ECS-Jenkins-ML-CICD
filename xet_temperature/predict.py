import json
import os
import sys

import joblib

from xet_temperature.metrics import current_metrics, read_real_temperature
from xet_temperature.train import MODEL_PATH, train


def estimate(metrics, model_path=MODEL_PATH):
    """Estimate temperature (C) for a dict of metrics."""
    bundle = joblib.load(model_path)
    row = [[metrics[name] for name in bundle["features"]]]
    return round(float(bundle["model"].predict(row)[0]), 1)


def main():
    if not os.path.exists(MODEL_PATH):
        train()
    metrics = current_metrics()
    real = read_real_temperature()
    estimated = estimate(metrics)
    report = {
        "metrics": metrics,
        "estimated_temperature_c": estimated,
        "real_sensor_temperature_c": real,
        "note": (
            "Estimated by an Extra Trees model trained on synthetic data. "
            "A real sensor value is shown only if the machine exposes one."
        ),
    }
    print(json.dumps(report, indent=2))
    with open("xet_temperature_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
