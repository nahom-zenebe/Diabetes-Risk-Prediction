from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "model" / "diabetes_pipeline.pkl"


model = joblib.load(model_path)

app = FastAPI(title="Diabetes Risk Prediction API")


origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000", 
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],           
)




class Patient(BaseModel):
    gender: int                     
    age: float
    hypertension: int             
    heart_disease: int             
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    smoking_history_current: int  
    smoking_history_ever: int
    smoking_history_former: int
    smoking_history_never: int
    smoking_history_not_current: int

@app.post("/predict")
def predict_diabetes(patient: Patient):
   
    input_df = pd.DataFrame([patient.dict()])
    
    
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    
 
    input_df = input_df[model.feature_names_in_]
    
   
    prob = model.predict_proba(input_df)[:,1][0]   
    pred_class = int(model.predict(input_df)[0]) 
    
    return {"prediction": pred_class, "probability": prob}
