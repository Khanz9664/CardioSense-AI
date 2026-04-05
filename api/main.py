import uuid
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import json
import uvicorn
from src.models.predict import HeartDiseasePredictor
from src.utils.logger import setup_logger

# Initialize Production-Grade Logger
logger = setup_logger("API-ENGINE")

app = FastAPI(
    title="CardioSense AI Inference Gateway",
    description="Production-grade clinical API for real-time risk assessment with traceability and audit logging.",
    version="2.1.0"
)

# Configuration & Loaders
MODEL_PATH = "models/heart_disease_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METADATA_PATH = "models/model_metadata.json"

predictor = None
metadata = None
model_version = "Unknown"

@app.on_event("startup")
def startup_event():
    global predictor, metadata, model_version
    logger.info("Initializing Clinical Intelligence System...")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        try:
            predictor = HeartDiseasePredictor(MODEL_PATH, PREPROCESSOR_PATH)
            logger.info("Predictor model and preprocessor successfully loaded.")
        except Exception as e:
            logger.error(f"Startup Failure: Could not load model artifacts. {e}")
    
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                model_version = metadata.get("version", "1.0.0")
            logger.info(f"Clinical metadata loaded. Active Model Version: {model_version}")
        except Exception as e:
            logger.error(f"Startup Warning: Could not parse metadata. {e}")

# --- PRODUCTION MIDDLEWARE ---

@app.middleware("http")
async def context_and_logging_middleware(request: Request, call_next):
    """
    Middleware to inject unique Request IDs and log request telemetry.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()
    
    # Process
    response = await call_next(request)
    
    # Finalize Telemetry
    duration = time.time() - start_time
    logger.info(f"REQ [{request_id}] | {request.method} {request.url.path} | STATUS: {response.status_code} | DUR: {duration:.4f}s")
    
    # Attach ID to Response for Traceability
    response.headers["X-Request-ID"] = request_id
    return response

# --- ERROR HANDLING MIDDLEWARE ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Intercepts all unhandled exceptions to prevent sensitive internal leakage
    and provide trace-ready error responses.
    """
    request_id = request.headers.get("X-Request-ID", "N/A")
    logger.error(f"EXCEPTION [{request_id}] | Unhandled error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Clinical Processing Error",
            "request_id": request_id,
            "message": "An unexpected failure occurred during inference. Refer to Request ID for auditing."
        }
    )

# --- API SCHEMAS ---

class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., description="1: Male, 0: Female")
    cp: int = Field(..., description="1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic")
    trestbps: int = Field(..., ge=80, le=200)
    chol: int = Field(..., ge=100, le=600)
    fbs: int = Field(..., description="1 for Blood Sugar > 120, 0 otherwise")
    restecg: int = Field(..., description="0: Normal, 1: ST-T wave abnormality, 2: LV Hypertrophy")
    thalach: int = Field(..., ge=60, le=220)
    exang: int = Field(..., description="1: Induced Angina, 0: otherwise")
    oldpeak: float = Field(..., ge=0.0, le=6.0)
    slope: int = Field(..., description="1: Upsloping, 2: Flat, 3: Downsloping")
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., description="3: Normal, 6: Fixed, 7: Reversable")

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "service": "CardioSense AI Inference Gateway",
        "status": "online",
        "version": app.version
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "model_version": model_version,
        "uptime_heartbeat": time.time()
    }

@app.post("/predict")
def predict_risk(request: Request, data: PatientData):
    """
    Executes real-time clinical risk prediction.
    """
    request_id = request.headers.get("X-Request-ID", "N/A")
    
    if not predictor:
        logger.warning(f"REQ [{request_id}] | Inference attempted while model is offline.")
        raise HTTPException(status_code=503, detail="Clinical Model is currently offline.")
    
    input_df = pd.DataFrame([data.dict()])
    prediction, probability = predictor.predict(input_df)
    
    result = {
        "prediction": int(prediction[0]),
        "risk_probability": round(float(probability[0][1]), 4),
        "status": "Positive (High Risk)" if prediction[0] == 1 else "Negative (Low Risk)",
        "model_version": model_version,
        "request_id": request_id
    }
    
    logger.info(f"REQ [{request_id}] | Inference Successful | Result: {result['status']} ({result['risk_probability']})")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
