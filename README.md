# Cats vs Dogs – End-to-End MLOps Pipeline

An end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) built using open-source tools including Git-LFS, MLflow, FastAPI, Docker, Docker Compose, and GitHub Actions.

This project demonstrates model development, experiment tracking, CI/CD automation, containerized deployment, monitoring, and post-deployment evaluation.

---

# Use Case

Binary image classification for a **Pet Adoption Platform** to automatically classify pet images as:

- Cat
- Dog

Dataset: Cats vs Dogs dataset from Kaggle  
Images resized to **224x224 RGB** for CNN compatibility.

---

# Tech Stack

| Component | Tool |
|------------|------|
| Language | Python 3.11 |
| Deep Learning | PyTorch |
| Experiment Tracking | MLflow |
| Data Versioning | Git-LFS |
| API | FastAPI |
| Containerization | Docker |
| Deployment | Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Monitoring | Custom in-app metrics + logging |

---

# Prerequisites

- Python **3.11**
- pip
- Docker
- Docker Compose (v2)
- Git
- Git-LFS
- Docker Hub account

---

# GitHub
### https://github.com/gowtham-potluri/pet-adoption-platform
### https://github.com/gowtham-potluri/pet-adoption-platform/actions/runs/22276094937

# Google Drive recording link
### https://drive.google.com/file/d/1isdWYEV6bI3cB_TZTo0P22raVBXiSaPf/view?usp=drive_link

### Note: Removed all the artifacts from the folder due to size restrictions


# Project Structure

```
pet-adoption-platform/
│
├── data/                  # Raw + processed dataset (Git-LFS tracked)
├── models/                # Trained model (.pt)
├── src/
│   ├── data/              # Preprocessing
│   ├── models/            # Training logic
│   ├── api/               # FastAPI inference service
│   └── utils/             # Performance evaluation
│
├── checks/smoke/          # Smoke tests
├── tests/                 # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/     # CI/CD pipelines
└── README.md
```

---

# M1: Model Development & Experiment Tracking

## Data & Code Versioning

- Source code versioned with Git
- Dataset versioned with Git-LFS

```bash
git lfs install
git lfs track "data/raw/**"
```

---

## Model Building

- Baseline CNN implemented in PyTorch
- Images resized to 224x224
- Data augmentation:
    - RandomHorizontalFlip
    - RandomRotation
- Train/Val/Test split: 80/10/10
- Model saved as:

```
models/model.pt
```

---

## Experiment Tracking (MLflow)

Logged:
- Training loss
- Validation accuracy
- Hyperparameters
- Model artifact
- Confusion matrix

Run locally:

## Steps:
### 1. python3.11 -m venv .venv
### 2. source .venv/bin/activate
### 3. pip install -r requirements.txt
### 4. python -m src.models.train
### 5. mlflow ui
### 6. docker build -t cats-dogs-api .
### 7. docker run -p 8080:8080 cats-dogs-api
### 8. naviate to localhost:8080/docs

```bash
mlflow ui
```

---

# M2: Model Packaging & Containerization

## Inference API (FastAPI)

### Health Check

```
GET /health
```

Response:
```json
{"status": "healthy"}
```

### Prediction

```
POST /predict
```

Response:
```json
{
  "label": "cat",
  "probability": 0.91
}
```

# Python Dependencies (Pinned Versions)

```txt
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.8.4
pillow==10.3.0
mlflow==2.12.1
fastapi==0.111.0
uvicorn==0.30.0
python-multipart==0.0.9
requests==2.31.0
pytest==8.2.0
```

---

## Dockerfile

Build image:

```bash
docker build -t cats-dogs-api .
```

Run container:

```bash
docker run -p 8080:8080 cats-dogs-api
```

---

## Docker Compose Deployment

```yaml
services:
  cats-dogs-api:
    image: gowthampotluri/cats-dogs-api:latest
    ports:
      - "8080:8080"
```

Run:

```bash
docker compose up -d
```

---

# M3: CI Pipeline (GitHub Actions)

Triggered on:
- Push
- Pull Request

Pipeline Steps:
1. Checkout repository
2. Install dependencies
3. Run unit tests (pytest)
4. Build Docker image
5. Push image to Docker Hub

Published Image:
```
gowthampotluri/cats-dogs-api:latest
```

---

# M4: CD Pipeline & Deployment

Triggered on push to `main`.

Deployment Steps:
1. Pull latest Docker image
2. Deploy via Docker Compose
3. Wait for service readiness
4. Run smoke tests
5. Fail pipeline if tests fail

---

# M5: Monitoring & Logging

## Logging

Logs:
```
METHOD PATH STATUS LATENCY
```

Example:
```
POST /predict - 200 - 52ms
GET /health - 200 - 3ms
```

View logs:

```bash
docker compose logs cats-dogs-api
```

---

### Performance Endpoint

```
GET /performance
```

Returns evaluation metrics on sample batch.

Response:
```json
{
  "accuracy": 1.0,
  "details": [
    {
      "file": "resources/cat1_test.jpg",
      "true_label": "cat",
      "pred_label": "cat",
      "probability": 0.09282832592725754,
      "latency_ms": 11.56,
      "correct": true
    },
    {
      "file": "resources/cat2_test.jpg",
      "true_label": "cat",
      "pred_label": "cat",
      "probability": 0.0004393496783450246,
      "latency_ms": 10.33,
      "correct": true
    },
    {
      "file": "resources/cat3_test.jpg",
      "true_label": "cat",
      "pred_label": "cat",
      "probability": 0.00010564936383161694,
      "latency_ms": 10.52,
      "correct": true
    },
    {
      "file": "resources/dog1_test.jpg",
      "true_label": "dog",
      "pred_label": "dog",
      "probability": 0.999234676361084,
      "latency_ms": 11.73,
      "correct": true
    },
    {
      "file": "resources/dog2_test.jpg",
      "true_label": "dog",
      "pred_label": "dog",
      "probability": 0.9999017715454102,
      "latency_ms": 10.56,
      "correct": true
    }
  ]
}
```

### Metrics Endpoint

```
GET /metrics
```

Returns:
- Total requests
- Last 1-hour requests
- Average latency

Response:
```json
{
  "last_1hr_requests": 10,
  "total_latency_ms": 419.63,
  "avg_latency_ms": 41.96
}
```

---

---

# Model Performance Comparison (Dataset Size Impact)

To evaluate how dataset size affects model generalization, the model was trained under two different dataset configurations.

## Experiment 1 — Small Dataset

- **Training Data:** 1,000 Cat images + 1,000 Dog images
- **Total Images:** 2,000
- **Accuracy:** `0.60`

This indicates moderate performance and limited generalization due to smaller dataset size.

---

## Experiment 2 — Full Dataset

- **Training Data:** 12,000 Cat images + 12,000 Dog images
- **Total Images:** 24,000
- **Accuracy:** `1.00`

This shows near-perfect classification performance on the evaluation dataset, demonstrating significant improvement with larger training data.

---

## Observations

| Dataset Size | Cats | Dogs | Total Images | Performance |
|-------------|------|------|--------------|-------------|
| Small       | 1,000 | 1,000 | 2,000 | 0.60 |
| Large       | 12,000 | 12,000 | 24,000 | 1.00 |

## Github Actions links:
### 0.6 Score
[![CI Pipeline](https://github.com/gowtham-potluri/pet-adoption-platform/actions/workflows/cicd.yaml/badge.svg)](https://github.com/gowtham-potluri/pet-adoption-platform/actions/runs/22272157522/job/64428974520)

### 1.0 Score
[![CI Pipeline](https://github.com/gowtham-potluri/pet-adoption-platform/actions/workflows/cicd.yaml/badge.svg)](https://github.com/gowtham-potluri/pet-adoption-platform/actions/runs/22276094937/job/64439733849)


### Key Insight

- Increasing dataset size dramatically improved model performance.
- Larger datasets help CNNs learn better feature representations.
- Data quantity has a direct impact on generalization capability.

---

## Conclusion

The experiment clearly demonstrates the importance of:
- Sufficient training data
- Proper dataset scaling
- Robust experiment tracking (via MLflow)

The `/performance` endpoint helps validate deployed model quality post-training and ensures production visibility into model accuracy.

---

# Run Locally (Complete Workflow)

### Clone

```bash
git clone <repo-url>
cd pet-adoption-platform
```

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python -m src.models.train
```

### Run API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
```

### Test

```bash
curl http://localhost:8080/health
```

---

# Demo Recording (Submission)

Show:
1. Code push to GitHub
2. CI pipeline execution
3. Docker image build & push
4. CD deployment
5. Smoke test success
6. `/predict` working
7. `/metrics` output
8. `/performance` output

---

# Deliverables

✔ Source code  
✔ requirements.txt  
✔ Dockerfile  
✔ docker-compose.yml  
✔ GitHub Actions CI/CD  
✔ Trained model (.pt)  
✔ Git-LFS dataset  
✔ Logs + Monitoring  
✔ Screen recording (<5 min)

---

# Summary

This project demonstrates:

- Reproducible ML training
- Experiment tracking
- Containerized inference
- Automated CI/CD
- Deployment validation
- Monitoring & logging
- Production-ready MLOps workflow
