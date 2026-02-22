import os
import time
import logging
from fastapi import FastAPI, UploadFile, File, Request
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