import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Tbilisi Apartment Price Prediction API")


MODEL_URI = "runs:/15d10aa9f51f48258ce9137cb84e7773/model"

model = mlflow.pyfunc.load_model(MODEL_URI)


class ApartmentInput(BaseModel):
    area: float
    rooms: int
    district: str
    floor: int
    condition: str
    parking: str


@app.get("/")
def home():
    return {"message": "Apartment Price Prediction API is running"}


@app.post("/predict")
def predict(data: ApartmentInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)

    return {
        "predicted_price": float(prediction[0])
    }