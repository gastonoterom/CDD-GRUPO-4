from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI
from joblib import load
from starlette.responses import FileResponse

app = FastAPI()


@dataclass
class PredictBody:
    population: int
    kidney: float
    meningitis: float
    cholera: float


@dataclass
class PredictResponse:
    respiratory_prediction: float


model = load('sandbox/data/rf_model.joblib')


@app.post("/predict")
def predict(body: PredictBody) -> PredictResponse:

    body_data = {
        'Chronic kidney disease': body.kidney / 10,
        'Meningitis': body.meningitis / 10,
        'Diarrheal diseases': body.cholera / 10,
    }

    df = pd.DataFrame([body_data])

    prediction = model.predict(df)[0]

    return PredictResponse(
        respiratory_prediction=prediction * 100,
    )


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
