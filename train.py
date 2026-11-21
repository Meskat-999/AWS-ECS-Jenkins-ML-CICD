import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.joblib"


def train(model_path=MODEL_PATH):
    """Train the model, save it, and return its test accuracy."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path} (test accuracy: {accuracy:.3f})")
    return accuracy


if __name__ == "__main__":
    train()
