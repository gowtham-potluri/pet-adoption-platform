import os
import yaml
import mlflow
import torch
import shutil
from src.data.preprocess import split_data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.models.model import SimpleCNN


def train():

    # ---------- Load Parameters ----------
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    learning_rate = params["train"]["learning_rate"]
    image_size = params["train"]["image_size"]

    # ---------- MLflow Setup ----------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cats-dogs-classification")

    # Ensure clean state
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():

        # Log parameters INSIDE run
        mlflow.log_params(params["train"])

        # ---------- Preprocessing ----------
        if os.path.exists("data/processed"):
            print("Removing old processed data")
            shutil.rmtree("data/processed")

        print("Running preprocessing")
        split_data("data/raw/PetImages", "data/processed")

        # ---------- Device ----------
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # ---------- Data ----------
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        train_data = datasets.ImageFolder("data/processed/train", transform=transform)
        val_data = datasets.ImageFolder("data/processed/val", transform=transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        # ---------- Model ----------
        model = SimpleCNN().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # ---------- Training ----------
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # ---------- Validation ----------
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = (outputs.cpu().numpy() > 0.5).astype(int)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.flatten())

        accuracy = accuracy_score(y_true, y_pred)
        mlflow.log_metric("val_accuracy", accuracy)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # ---------- Confusion Matrix ----------
        cm = confusion_matrix(y_true, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")

        mlflow.log_artifact("confusion_matrix.png")

        # ---------- Save Model ----------
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model.pt")
        mlflow.log_artifact("models/model.pt")

        print("Training complete and model saved.")

if __name__ == "__main__":
    train()