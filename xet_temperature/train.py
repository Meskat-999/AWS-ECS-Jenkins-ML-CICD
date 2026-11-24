import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from xet_temperature.metrics import FEATURES

MODEL_PATH = "xet_temperature_model.joblib"


def make_synthetic_data(n_samples=3000, seed=42):
    """Synthetic server data: columns follow metrics.FEATURES, plus temperature."""
    rng = np.random.default_rng(seed)
    cpu = rng.uniform(0, 100, n_samples)
    mem = rng.uniform(10, 90, n_samples)
    load = rng.uniform(0, 8, n_samples)
    noise = rng.normal(0, 1.5, n_samples)
    temperature = 35 + 0.45 * cpu + 0.05 * mem + 2.0 * load + noise
    x = np.column_stack([cpu, mem, load])
    return x, temperature


def train(model_path=MODEL_PATH):
    """Train the model, save it, and return (mae, r2) on a held-out test set."""
    x, y = make_synthetic_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    joblib.dump({"model": model, "features": FEATURES}, model_path)
    print(f"Extra Trees model saved to {model_path}")
    print(f"Test MAE: {mae:.2f} C | R2: {r2:.3f}")
    return mae, r2


if __name__ == "__main__":
    train()
