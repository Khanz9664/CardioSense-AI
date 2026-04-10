import uuid
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import json
import uvicorn
from src.models.predict import HeartDiseasePredictor
from src.utils.logger import setup_logger
from src.monitoring.logger import MonitoringLogger
from src.monitoring.engine import MonitoringEngine
from src.utils.version_utils import get_model_version
from fastapi import BackgroundTasks

# Initialize Production-Grade Logger
logger = setup_logger("API-ENGINE")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for initializing clinical model artifacts
    and cleaning up resources on shutdown.
    """
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
                model_version = metadata.get("version", "2.4.0")
                app.version = model_version # Sync FastAPI version if metadata updated
            logger.info(f"Clinical metadata loaded. Active Model Version: {model_version}")
        except Exception as e:
            logger.error(f"Startup Warning: Could not parse metadata. {e}")
            
    yield
    # Cleanup logic (if needed) can go here
    logger.info("Shutting down Clinical Intelligence System...")

app = FastAPI(
    title="CardioSense AI: Clinical Decision Support API",
    description="Medical-grade cardiovascular risk stratification engine with integrated ACC/AHA safety guardrails and multi-modal explainability.",
    version=get_model_version(),
    lifespan=lifespan
)

# Configuration & Loaders
MODEL_PATH = "models/heart_disease_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METADATA_PATH = "models/model_metadata.json"

predictor = None
metadata = None
model_version = "Unknown"
mon_logger = MonitoringLogger()
mon_engine = MonitoringEngine()


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
def predict_risk(request: Request, data: PatientData, background_tasks: BackgroundTasks):
    """
    Executes real-time clinical risk prediction with background monitoring.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    if not predictor:
        logger.warning(f"REQ [{request_id}] | Inference attempted while model is offline.")
        raise HTTPException(status_code=503, detail="Clinical Model is currently offline.")
    
    input_df = pd.DataFrame([data.dict()])
    prediction, probability = predictor.predict(input_df)
    
    prob_val = round(float(probability[0][1]), 4)
    result = {
        "prediction": int(prediction[0]),
        "risk_probability": prob_val,
        "status": "Positive (High Risk)" if prediction[0] == 1 else "Negative (Low Risk)",
        "model_version": model_version,
        "request_id": request_id
    }
    
    # Async Persistence for Drift Monitoring
    background_tasks.add_task(
        mon_logger.log_prediction, 
        request_id, input_df, prediction[0], prob_val, model_version
    )
    
    logger.info(f"REQ [{request_id}] | Inference Successful | Result: {result['status']}")
    return result

@app.post("/feedback/{request_id}")
def submit_feedback(request_id: str, actual_outcome: int):
    """
    Clinician endpoint to submit ground truth outcome (0: Healthy, 1: Disease) 
    for Concept Drift monitoring.
    """
    try:
        mon_logger.log_feedback(request_id, actual_outcome)
        return {"status": "Feedback recorded", "request_id": request_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status")
def get_monitoring_status():
    """
    Returns a high-level summary of data and concept drift.
    """
    drift_stats = mon_engine.run_drift_analysis(window_size=100)
    perf_stats = mon_engine.run_performance_audit()
    
    return {
        "drift": drift_stats,
        "performance": perf_stats,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
