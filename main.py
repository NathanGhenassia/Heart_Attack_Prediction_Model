from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
import joblib


app = FastAPI(title="Deploy-heart-attack-model", version="0.0.1")


# -----------------------------------------
# LOAD AI MODEL
# -----------------------------------------
model = joblib.load("model/logistic_regression_model_heart_attack_v01.pkl")


@app.post("/api/v1/predict-heart-attack", tags=["heart-attack"])
async def predict(
    age: float,
    sex: float,
    cp: float,
    trtbps: float,
    chol: float,
    fbs: float,
    restecg: float,
    thalachh: float,
    exng: float,
    oldpeak: float,
    slp: float,
    caa: float,
    thall: float,
):
    dictionary = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall,
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        result = "More chance of heart attack" if prediction[0] == 1 else "Less chance of heart attack"
        return JSONResponse(status_code=status.HTTP_200_OK, content={"prediction": result})

    except Exception as e:
        raise HTTPException(detail=str(e), status_code=status.HTTP_400_BAD_REQUEST)
