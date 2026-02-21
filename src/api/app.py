from fastapi import FastAPI, UploadFile, File
from src.api.inference import predict

app = FastAPI(title="Cats vs Dogs Classifier")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result