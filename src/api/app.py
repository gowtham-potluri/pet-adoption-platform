import time
import logging
from fastapi import FastAPI, UploadFile, File, Request
from src.api.inference import predict

app = FastAPI(title="Cats vs Dogs Classifier")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_logger")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result