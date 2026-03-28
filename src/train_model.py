import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Base project path
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
TRAIN_DATA_PATH = BASE_DIR / "data" / "machine_failure_train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


def main():
    print("Loading training data...")
    df = pd.read_csv(TRAIN_DATA_PATH)

    print("Training data loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # Features and target
    X = df.drop(columns=["machine_id", "failure"])
    y = df["failure"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully at: {MODEL_PATH}")


if __name__ == "__main__":
    main()