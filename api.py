from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

model = joblib.load("best_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

class PlayerFeatures(BaseModel):
    Pos1: int
    Age: float
    Tm: int
    G: float
    GS: float
    MP: float
    FG: float
    FGA: float
    FG_dot: float = Field(alias="FG.")
    X3P: float
    X3PA: float
    X3P_dot: float = Field(alias="X3P.")
    X2P: float
    X2PA: float
    X2P_dot: float = Field(alias="X2P.")
    eFG_dot: float = Field(alias="eFG.")
    FT: float
    FTA: float
    FT_dot: float = Field(alias="FT.")
    ORB: float
    DRB: float
    TRB: float
    AST: float
    STL: float
    BLK: float
    TOV: float
    PF: float
    PTS: float
    Salary: float
    mean_views: float
    Conference: int
    Role: int
    Fvot: float
    FRank: float
    Pvot: float
    PRank: float
    Mvot: float
    MRank: float
    Score: float

app = FastAPI()

@app.post("/predict/")
def predict(data: PlayerFeatures):
    try:
        input_df = pd.DataFrame([data.dict(by_alias=True)])
        scaled_input = scaler.transform(input_df)
        prob = model.predict_proba(scaled_input)[0][1]
        prediction = "All-Star" if prob >= 0.5 else "Not All-Star"

        return {
            "all_star_probability": float(round(prob * 100, 2)), 
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
