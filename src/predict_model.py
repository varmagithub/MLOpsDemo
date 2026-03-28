import pandas as pd
import joblib
from pathlib import Path


# Base project path
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
PREDICT_DATA_PATH = BASE_DIR / "data" / "machine_failure_predict.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


# --------------------------------------
# Reusable prediction function (for API)
# --------------------------------------
def predict(input_data: pd.DataFrame):
    """
    Takes input DataFrame and returns prediction labels
    """
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(input_data)

    prediction_labels = [
        "Failure Risk" if pred == 1 else "Normal Operation"
        for pred in predictions
    ]

    return prediction_labels


# --------------------------------------
# Batch prediction (Day 02 usage)
# --------------------------------------
def main():
    print("Loading prediction data...")
    df = pd.read_csv(PREDICT_DATA_PATH)

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    # Keep machine_id separately for output
    machine_ids = df["machine_id"]

    # Feature columns only
    X_input = df.drop(columns=["machine_id"])

    print("Running predictions...")
    predictions = model.predict(X_input)

    # Map prediction values to readable labels
    prediction_labels = [
        "Failure Risk" if pred == 1 else "Normal Operation"
        for pred in predictions
    ]

    # Create result dataframe
    results = pd.DataFrame({
        "machine_id": machine_ids,
        "prediction": prediction_labels
    })

    print("\nPrediction Results:")
    print(results)

    # Optional: save results
    output_path = BASE_DIR / "models" / "prediction_output.csv"
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved at: {output_path}")


# --------------------------------------
# Entry point
# --------------------------------------
if __name__ == "__main__":
    main()