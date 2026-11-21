import joblib
import numpy as np
from flask import Flask, jsonify, request

from train import MODEL_PATH

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    """Health check / landing route (ALB health checks hit this)."""
    return jsonify(status="ok", message="Iris model API is running")


@app.route("/predict", methods=["POST"])
def predict():
    """Expects JSON: {"features": [sepal_len, sepal_wid, petal_len, petal_wid]}."""
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify(error="JSON body with 'features' is required"), 400

    features = data["features"]
    if not isinstance(features, list) or len(features) != 4:
        return jsonify(error="'features' must be a list of 4 numbers"), 400

    try:
        x = np.array(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify(error="'features' must contain only numbers"), 400

    prediction = int(model.predict(x)[0])
    return jsonify(prediction=prediction, species=CLASS_NAMES[prediction])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
