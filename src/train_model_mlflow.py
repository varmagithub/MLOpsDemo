import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "machine_failure_train.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["machine_id", "failure"])
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("Machine_Failure_Prediction")

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        print("Training model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        joblib.dump(model, MODEL_PATH)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 8)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        cm_path = ARTIFACT_DIR / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        report_path = ARTIFACT_DIR / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))

        print("MLflow tracking completed successfully.")


if __name__ == "__main__":
    main()