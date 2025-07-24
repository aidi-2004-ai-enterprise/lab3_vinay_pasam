from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import xgboost as xgb
import pickle
import os

app = FastAPI()

# Paths relative to main.py inside app/
MODEL_PATH = "app/data/xgb_penguin_model.json"
LABEL_ENCODER_PATH = "app/data/label_encoder.pkl"

# Load the trained XGBoost model
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please run train.py to generate it.")

model = xgb.Booster()
model.load_model(MODEL_PATH)

# Load label encoder
if not os.path.exists(LABEL_ENCODER_PATH):
    raise RuntimeError(f"Label encoder file '{LABEL_ENCODER_PATH}' not found. Please run train.py to generate it.")

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Enums for strict validation
class SexEnum(str, Enum):
    Male = "Male"
    Female = "Female"

class IslandEnum(str, Enum):
    Biscoe = "Biscoe"
    Dream = "Dream"
    Torgersen = "Torgersen"

# Input schema with Enum fields
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int  # Will be dropped before prediction
    sex: SexEnum
    island: IslandEnum

expected_cols = [
    'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
    'sex_Female', 'sex_Male',
    'island_Biscoe', 'island_Dream', 'island_Torgersen'
]

@app.post("/predict")
def predict_species(features: PenguinFeatures):
    try:
        df = pd.DataFrame([features.dict()])
        df = df.drop('year', axis=1)
        df = pd.get_dummies(df, columns=['sex', 'island'])

        # Add missing columns with zeros
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]
        dmatrix = xgb.DMatrix(df)

        pred_numeric = model.predict(dmatrix)[0]
        pred_species = le.inverse_transform([int(pred_numeric)])[0]

        return {"predicted_species": pred_species}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
