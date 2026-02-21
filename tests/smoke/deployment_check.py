import requests
import sys

BASE_URL = "http://localhost:8080"

# Health check
try:
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print("Health check passed")
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)

# Prediction check
sample_input = {"image_path": "tests/sample/cat.jpg"}
try:
    r = requests.post(f"{BASE_URL}/predict", json=sample_input)
    assert r.status_code == 200
    print("Prediction endpoint passed:", r.json())
except Exception as e:
    print(f"Prediction endpoint failed: {e}")
    sys.exit(1)