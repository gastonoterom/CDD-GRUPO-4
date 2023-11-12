from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI
from joblib import load
from starlette.responses import FileResponse

app = FastAPI()


@dataclass
class PredictBody:
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
        'Chronic kidney disease': body.kidney,
        'Meningitis': body.meningitis,
        'Diarrheal diseases': body.cholera,
    }

    # Create a DataFrame using the body_data
    df = pd.DataFrame([body_data])

    prediction = model.predict(df)

    return PredictResponse(
        respiratory_prediction=prediction[0],
    )


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
