import torch
from torchvision import transforms
from PIL import Image
import io

from src.models.model import SimpleCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model once
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load("models/model.pt", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_bytes: bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probability = output.item()
        label = "dog" if probability > 0.5 else "cat"

    return {
        "label": label,
        "probability": float(probability)
    }