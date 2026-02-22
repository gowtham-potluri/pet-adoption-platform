import requests
from sklearn.metrics import accuracy_score

BASE_URL = "http://localhost:8080"

# Example test data (simulate real requests)
test_batch = [
    {"image_path": "checks/smoke/cat_test_image.jpg", "label": "cat"},
    {"image_path": "checks/smoke/dog_test_image.jpg", "label": "dog"},
]

y_true = []
y_pred = []

for item in test_batch:
    # Open the image file and upload it
    with open(item["image_path"], "rb") as f:
        files = {"file": f}
        r = requests.post(f"{BASE_URL}/predict", files=files)
        r.raise_for_status()  # fail immediately if HTTP error
        pred = r.json()["predicted_class"]

    y_true.append(item["label"])
    y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
print(f"Performance on test batch: Accuracy = {acc:.4f}")