# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI()
model = joblib.load("fraud_detection_model.pkl")

class Transaction(BaseModel):
    features: list  # [value1, value2, ..., valueN]

class InputData(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
