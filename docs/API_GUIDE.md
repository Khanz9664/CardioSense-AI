# Production API Guide: CardioSense AI

### Production Interface (FastAPI)

![FastAPI Interactive Documentation](../app/assets/App_Screenshots/API_Docs.png)

CardioSense AI exposes a production-grade FastAPI REST interface for seamless integration with Electronic Health Record (EHR) and Hospital Management Systems.

---

## 1. Base URL & Endpoints

- **Development**: `http://localhost:8000`
- **Production**: (Based on deployment)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/predict` | `POST` | Primary inference endpoint for patient risk assessment. |
| `/metadata` | `GET` | Returns model version, accuracy, and healthy benchmarks. |
| `/health` | `GET` | System health check and clinical engine versioning. |
| `/docs` | `GET` | Interactive Swagger UI. |

---

## 2. API Reference

### POST `/predict`
Submit patient vitals to receive a heart disease risk probability.

**Injected Headers**:
- `X-Request-ID`: A unique UUID for audit tracing (e.g., `550e8400-e29b-411d-a716-446655440000`).

**Request Body (JSON)**:
```json
{
  "age": 55,
  "sex": 1,
  "cp": 4,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 130,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 2,
  "ca": 1,
  "thal": 7
}
```

**Response (JSON)**:
```json
{
    "prediction": 1,
    "risk_probability": 0.9234,
    "status": "High Risk",
    "version": "2.1.0",
    "request_id": "..."
}
```

---

### GET `/health`
Check if the system is online and the clinical engine is correctly loaded.

**Response (JSON)**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "2.1.0",
    "timestamp": "..."
}
```

---

## 3. Production Features

### Clinical Auditability
Every request is assigned a unique `X-Request-ID`. This ID is returned in the response headers and logged alongside the inference results, enabling full downstream traceability for clinical audits.

### Structured Logging
CardioSense AI uses a rotating file-based JSON logger (`logs/cardiosense.log`). 
- **Rotation**: 5MB per file, max 3 backups.
- **Content**: API access logs, inference probability distribution, and internal trace IDs.

---

## 4. Implementation Examples

### Python Integration (with Tracing)
```python
import requests
import uuid

request_id = str(uuid.uuid4())
headers = {"X-Request-ID": request_id}

patient_data = { ... }

response = requests.post("http://localhost:8000/predict", json=patient_data, headers=headers)
print(f"Audit ID: {response.headers.get('X-Request-ID')}")
print(response.json())
```

---

## 5. Error & Status Codes

- `200 OK`: Success.
- `400 Bad Request`: Input validation failed or clinical logic error.
- `503 Service Unavailable`: Clinical engine weights not found.
- `500 Internal Error`: Global middleware caught an unhandled exception (stack traces are suppressed for security).
