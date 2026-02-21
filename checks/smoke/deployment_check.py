import requests
import sys
import time

BASE_URL = "http://localhost:8080"
SAMPLE_IMAGE = "checks/smoke/cat_test_image.jpg"

# Wait a few seconds for the server to start
print("Waiting for the API server to be ready...")
time.sleep(5)  # adjust as needed

# Health check
try:
    r = requests.get(f"{BASE_URL}/health")
    r.raise_for_status()
    print("Health check passed")
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)

# Prediction check (uploading file)
try:
    with open(SAMPLE_IMAGE, "rb") as f:
        files = {"file": f}
        r = requests.post(f"{BASE_URL}/predict", files=files)
        r.raise_for_status()
        print("Prediction endpoint passed:", r.json())
except Exception as e:
    print(f"Prediction endpoint failed: {e}")
    sys.exit(1)