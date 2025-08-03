# Fraud Detection API (MLOps-ready)

This project provides a fully containerized machine learning-based REST API for detecting fraudulent transactions. It integrates data preprocessing, a trained `RandomForestClassifier`, API deployment via Flask, and retraining logic based on data drift — all prepared for production environments using Docker and Conda.

---

## Features

- REST API for fraud detection via `/predict` endpoint
- Pre-trained ML model using scikit-learn
- Drift detection and retraining logic
- Dockerized for portability
- Conda-based reproducible environment
- MLflow logging and tracking
- Works with cURL, Postman, or custom clients

---

## Getting Started

### Clone this repo

```bash
git clone https://github.com/Phips92/Fraud_detector
cd fraud-detection-api
````

### Build the Docker image

```bash
docker build -t fraud-api-conda .
```

### Run the container

```bash
docker run -d -p 5000:5000 fraud-api-conda
```

---

## API Usage

### Authentication

The API is secured with a simple header-based key.

```http
Header: x-api-key: geht_di_nix_an
```

### POST `/predict`

* **URL:** `http://localhost:5000/predict`
* **Method:** `POST`
* **Content-Type:** `application/json`
* **Header:** `x-api-key: geht_di_nix_an`

#### Request Example

```json
{
  "amt": 120.50,
  "city_pop": 35000,
  "merch_lat": 40.75,
  "merch_long": -73.99,
  "merchant": "fraud_Kirlin and Sons",
  "dob": "1980-05-04",
  "trans_date_trans_time": "2020-12-01T14:30:00"
}
```

#### Response Example

```json
{
  "fraud_probability": 0.0273
}
```

---

## Architecture Overview

The project includes the following core modules:

| File                        | Purpose                                 |
| --------------------------- | --------------------------------------- |
| `model.py`                  | Training logic + MLflow logging         |
| `evaluate_model.py`         | Performance evaluation of new data      |
| `generate_data.py`          | Simulates time-based data drift         |
| `detect_drift_and_train.py` | Drift detection and retraining pipeline |
| `app.py`                    | Flask API for real-time predictions     |
| `Dockerfile`, `env.yml`     | Docker + Conda environment definition   |

---

## Development & Monitoring

### MLflow Tracking

All training runs and metrics are tracked locally in:

```
mlruns/
└── fraud_detection/
```

Start the MLflow UI locally with:

```bash
mlflow ui
```

---

### Drift Simulation & Retraining

To simulate monthly data drift and evaluate/retrain the model:

```bash
python generate_data.py
python detect_drift_and_train.py
```

---

## Testing

### Local request via cURL:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: geht_di_nix_an" \
  -d @Request.json
```

### Using Postman:

1. Import the `Request.json` file as a request
2. Set header: `x-api-key: geht_di_nix_an`
3. Set method: `POST`
4. Target URL: `http://localhost:5000/predict`

---

## Project Setup Details

| Setting        | Value                                       |
| -------------- | ------------------------------------------- |
| Python Version | 3.10                                        |
| Environment    | Conda (via `environment.yml`)               |
| Key Libraries  | scikit-learn, Flask, MLflow, pandas, joblib |
| Base Image     | `continuumio/miniconda3`                    |
| Port           | 5000 (Flask API)                            |

---

## Project Structure

```
├── app.py
├── model.py
├── evaluate_model.py
├── detect_drift_and_train.py
├── generate_data.py
├── Request.json
├── Dockerfile
├── environment.yml
├── requirements.txt
├── fraud_model.joblib
├── scaler.joblib
├── merchant_columns.joblib
└── data/
    ├── month_01.csv
    └── ...
```

---

## License & Author

**Author**: \[Philipp Mc Guire]

**License**: \This project is licensed under the GNU General Public License v3.0


