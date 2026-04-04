from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import json
import uvicorn
from src.models.predict import HeartDiseasePredictor

app = FastAPI(
    title="Cardiovascular Risk Prediction API",
    description="Production-grade API for real-time heart disease risk assessment.",
    version="1.0.0"
)

# Load the predictor
MODEL_PATH = "models/heart_disease_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METADATA_PATH = "models/model_metadata.json"

predictor = None
metadata = None

@app.on_event("startup")
def startup_event():
    global predictor, metadata
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        predictor = HeartDiseasePredictor(MODEL_PATH, PREPROCESSOR_PATH)
    
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

# Pydantic Schema for Input Validation
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., description="1 for Male, 0 for Female")
    cp: int = Field(..., description="1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic")
    trestbps: int = Field(..., ge=80, le=200)
    chol: int = Field(..., ge=100, le=600)
    fbs: int = Field(..., description="1 for Blood Sugar > 120, 0 otherwise")
    restecg: int = Field(..., description="0: Normal, 1: ST-T wave abnormality, 2: LV Hypertrophy")
    thalach: int = Field(..., ge=60, le=220)
    exang: int = Field(..., description="1 for Exercise Induced Angina, 0 otherwise")
    oldpeak: float = Field(..., ge=0.0, le=6.0)
    slope: int = Field(..., description="1: Upsloping, 2: Flat, 3: Downsloping")
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., description="3: Normal, 6: Fixed defect, 7: Reversable defect")

@app.get("/")
def read_root():
    return {"message": "Heart Disease Risk Prediction API is running."}

@app.get("/health")
def health_check():
    if predictor:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

@app.get("/metadata")
def get_metadata():
    if metadata:
        return metadata
    raise HTTPException(status_code=404, detail="Model metadata not found.")

@app.post("/predict")
def predict_risk(data: PatientData):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized. Run training first.")
    
    try:
        # Convert Pydantic model to DataFrame for the predictor
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Inference
        prediction, probability = predictor.predict(input_df)
        
        # Format response
        result = {
            "prediction": int(prediction[0]),
            "risk_probability": float(probability[0][1]),
            "status": "High Risk" if prediction[0] == 1 else "Low Risk"
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
