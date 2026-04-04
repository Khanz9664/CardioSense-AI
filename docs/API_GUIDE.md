# Production API Guide: CardioSense AI

CardioSense AI exposes a production-grade FastAPI REST interface for seamless integration with Electronic Health Record (EHR) and Hospital Management Systems.

---

## 1. Base URL & Endpoints

- **Development**: `http://localhost:8000`
- **Production**: (Based on deployment)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/predict` | `POST` | Primary inference endpoint for patient risk assessment. |
| `/metadata` | `GET` | Returns model version, accuracy, and healthy benchmarks. |
| `/health` | `GET` | System health check and model loading status. |
| `/docs` | `GET` | Interactive Swagger UI. |

---

## 2. API Reference

### POST `/predict`
Submit patient vitals to receive a heart disease risk probability.

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
    "status": "High Risk"
}
```

---

### GET `/metadata`
Retrieves model performance metrics and training metadata.

**Response (JSON)**:
```json
{
    "accuracy": 0.9016,
    "roc_auc": 0.9418,
    "best_params": { ... },
    "healthy_baseline": { ... }
}
```

---

### GET `/health`
Check if the system is online and the model is correctly loaded.

**Response (JSON)**:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

---

## 3. Implementation Examples

### Python Integration
```python
import requests

patient_data = {
    "age": 60, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 2, "ca": 0, "thal": 3
}

response = requests.post("http://localhost:8000/predict", json=patient_data)
print(response.json())
```

### CURL
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 50,
  "sex": 1,
  "cp": 3,
  "trestbps": 120,
  "chol": 230,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 3
}'
```

---

## 4. Error Codes

- `400 Bad Request`: Input validation failed (e.g., `age` out of range).
- `503 Service Unavailable`: Model files not found; run training pipeline first.
- `500 Internal Server Error`: Unexpected system failure.
