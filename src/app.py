from fastapi import FastAPI, HTTPException
import pandas as pd
from src.predict_model import predict

app = FastAPI(title="Machine Failure Prediction API")


@app.get("/")
def home():
    return {"message": "Machine Failure Prediction API is running"}


@app.post("/predict")
def predict_single(data: dict):
    try:
        input_df = pd.DataFrame([data])
        result = predict(input_df)

        return {
            "prediction": result[0]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))