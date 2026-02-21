import requests
from sklearn.metrics import accuracy_score

BASE_URL = "http://localhost:8080"

# Example test data (simulate real requests)
test_batch = [
    {"image_path": "checks/smoke/cat_test_image.jpg", "label": 0},
    {"image_path": "checks/smoke/dog_test_image.jpg", "label": 1},
]

y_true = []
y_pred = []

for item in test_batch:
    r = requests.post(f"{BASE_URL}/predict", json={"image_path": item["image_path"]})
    pred = r.json()["predicted_class"]
    y_true.append(item["label"])
    y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
print(f"Post-deployment accuracy: {acc:.4f}")