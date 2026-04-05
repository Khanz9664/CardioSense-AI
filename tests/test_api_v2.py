import pytest
from fastapi.testclient import TestClient
from api.main import app
import uuid

# Use the app from api/main
client = TestClient(app)

def test_root_endpoint_metadata():
    # Test root returns professional clinical name and version
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "CardioSense AI" in data["service"]
    assert "v" in data["version"] or "." in data["version"]

def test_request_id_traceability():
    # Test that X-Request-ID is injected into responses
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    # Verify it is a valid UUID
    uuid.UUID(request_id)

def test_health_check_lineage():
    # Test health check returns model version for clinical auditing
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "status" in data

def test_predict_validation_edge_cases():
    # CASE 1: Age out of range (Negative)
    bad_age = {
        "age": -1, "sex": 1, "cp": 1, "trestbps": 120, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    }
    response = client.post("/predict", json=bad_age)
    assert response.status_code == 422 # Pydantic Validation Error
    
    # CASE 2: BP out of clinical range (> 200)
    bad_bp = bad_age.copy()
    bad_bp["age"] = 50
    bad_bp["trestbps"] = 250
    response = client.post("/predict", json=bad_bp)
    assert response.status_code == 422
    
    # CASE 3: Valid Input
    valid_data = bad_age.copy()
    valid_data["age"] = 50
    valid_data["trestbps"] = 120
    # NOTE: Since predictor might be None in test environment without artifacts, 
    # we check for 200 OR 503 (Model offline) but NOT 422 (Validation)
    response = client.post("/predict", json=valid_data)
    assert response.status_code in [200, 503]

def test_error_middleware_exception_handling():
    # Deliberately cause a failure (by passing None where not allowed if possible, or mocking)
    # Here we just verify that a completely malformed body (not JSON) returns a sensible 400
    response = client.post("/predict", content="not-a-json")
    assert response.status_code == 422 # FastAPI built-in handling for non-JSON
