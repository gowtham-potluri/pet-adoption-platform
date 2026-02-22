import os
import time
import requests
import logging
from collections import deque
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Request
from sklearn.metrics import accuracy_score
from src.api.inference import predict

app = FastAPI(title="Cats vs Dogs Classifier")

# Set up logging
# Ensure logs folder exists
os.makedirs("/app/logs", exist_ok=True)

logging.basicConfig(
    filename="/app/logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("api_logger")
logger.info("API service starting")

# Store requests as (timestamp, latency_in_ms)
request_log = deque()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000  # in milliseconds

    # Store timestamp and latency
    request_log.append((datetime.utcnow(), latency))

    # Optional: prune entries older than 1 hour
    cutoff = datetime.utcnow() - timedelta(hours=1)
    while request_log and request_log[0][0] < cutoff:
        request_log.popleft()

    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time  # in seconds

    # Convert to ms if less than 1 second
    if latency < 1.0:
        latency_str = f"{latency * 1000:.0f}ms"
    else:
        latency_str = f"{latency:.2f}s"

    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {latency_str}")
    return response

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result

@app.get("/performance")
async def performance_endpoint():
    """
    Evaluate model performance on a small test batch.
    Returns accuracy, predictions, and latency per sample.
    """
    # Example test data (can be replaced with real test dataset)
    test_batch = [
        {"image_path": "resources/cat1_test.jpg", "label": "cat"},
        {"image_path": "resources/cat2_test.jpg", "label": "cat"},
        {"image_path": "resources/cat3_test.jpg", "label": "cat"},
        {"image_path": "resources/dog1_test.jpg", "label": "dog"},
        {"image_path": "resources/dog2_test.jpg", "label": "dog"},
    ]

    results = []
    y_true = []
    y_pred = []

    for item in test_batch:
        start_time = time.time()
        try:
            with open(item["image_path"], "rb") as f:
                image_bytes = f.read()
                pred_result = predict(image_bytes)
                pred_label = pred_result.get("label")
                pred_prob = pred_result.get("probability")

            latency = time.time() - start_time

            y_true.append(item["label"])
            y_pred.append(pred_label)

            # Collect per-sample results
            results.append(
                {
                    "file": item["image_path"],
                    "true_label": item["label"],
                    "pred_label": pred_label,
                    "probability": pred_prob,
                    "latency_ms": round(latency * 1000, 2),
                    "correct": pred_label == item["label"],
                }
            )

            logger.info(
                f"{item['image_path']} | True: {item['label']} | Pred: {pred_label} | "
                f"Prob: {pred_prob} | Latency: {latency*1000:.1f}ms"
            )

        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                f"Failed to predict {item['image_path']} | Error: {e} | "
                f"Latency: {latency*1000:.1f}ms"
            )
            results.append(
                {"file": item["image_path"], "error": str(e), "latency_ms": round(latency * 1000, 2)}
            )

    # Compute overall accuracy
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return {"accuracy": acc, "details": results}

@app.get("/metrics")
def metrics():
    """
    Returns request count and total latency for the last 1 hour
    """
    cutoff = datetime.utcnow() - timedelta(hours=1)
    recent_requests = [(ts, lat) for ts, lat in request_log if ts >= cutoff]

    total_requests = len(recent_requests)
    total_latency = sum(lat for ts, lat in recent_requests)  # in ms

    return {
        "last_1hr_requests": total_requests,
        "total_latency_ms": round(total_latency, 2),
        "avg_latency_ms": round(total_latency / total_requests, 2) if total_requests else 0
    }