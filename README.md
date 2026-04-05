# Fraud Nirikshak: Credit Card Fraud Detection Pipeline

![FraudGuard ML](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-00a393)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E)

## 1. Project Overview

Financial institutions process millions of transactions daily, making manual fraud detection impossible. **Fraud Nirikshak** is an end-to-end Machine Learning system designed to identify fraudulent credit card transactions in real-time. 

The core challenge in this domain is **extreme class imbalance** (typically <0.2% of transactions are fraudulent), demanding strict evaluation metrics, robust resampling techniques, and careful threshold tuning. This project moves beyond a simple Jupyter Notebook model by wrapping the ML pipeline in a robust FastAPI backend and providing an interactive frontend UI to demonstrate real-time inference, model confidence, and system resilience.

---

## 2. Dataset Description

The system is trained on a highly anonymized dataset of European credit card transactions.

* **V1–V28**: Due to confidentiality issues, the original features (like merchant category, location, etc.) are hidden. Instead, they are the principal components obtained via PCA (Principal Component Analysis).
* **Amount**: The transaction amount in USD. Unlike the V-features, this is not PCA-transformed and requires explicit scaling handling during training.
* **Time**: The seconds elapsed between the given transaction and the first transaction in the dataset.
* **Class Imbalance**: The dataset is intensely skewed. Fraud accounts for ~0.17% of all transactions, establishing a strong baseline where a "dumb" model guessing "Legitimate" 100% of the time would still achieve 99.83% accuracy.

---

## 3. Machine Learning Pipeline

### Data Processing
Data normalization is critical. The PCA-transformed features (V1-V28) are inherently scaled, but the `Amount` feature is heavily skewed and requires robust scaling (e.g., `RobustScaler`) to mitigate the impact of outliers. 
To address the severe class imbalance during training, we utilize **SMOTE (Synthetic Minority Over-sampling Technique)**. This injects synthetic examples of the minority class into the training space, allowing the model to learn fraud patterns rather than just defaulting to the majority class.

### Model Selection
Multiple models were evaluated, prioritizing **Logistic Regression** and **Random Forests**.
* **Why Random Forest was selected**: The relationship between PCA features and fraud is highly non-linear. Random Forests construct diverse decision trees inherently resilient to overfitting, capturing the complex, multi-dimensional boundary separating legit from fraud better than linear models.
* **Why PR-AUC**: Standard ROC-AUC is misleading on highly imbalanced datasets because the large number of true negatives artificially inflates the score. We used **Precision-Recall AUC (PR-AUC)** to optimize explicitly for the minority class.

### Threshold Tuning
A standard model threshold is `0.5`, but in fraud detection, false negatives (missed fraud) are significantly costlier than false positives (a temporary block).
* We employed a cross-validation approach to test thresholds across precision-recall pairs.
* The operating threshold was dynamically tuned (e.g., to `~0.39`) to heavily penalize missing fraud while retaining an acceptable precision level, ensuring users aren't overwhelmed by false alarms.

### Calibration
Random Forests do not output true mathematical probabilities; they output the fraction of voting trees. This means the outputs are often aggressively clustered and uncalibrated.
* We wrapped the estimator using `CalibratedClassifierCV` (using Isotonic or Platt scaling). This ensures that when the API returns a `36%` fraud probability, it accurately reflects the real-world statistical likelihood of fraud.

### Key Learning: Out-of-Distribution (OOD)
During testing, we discovered the **OOD problem**. If a user inputs random, extreme values into the UI (e.g., $9,999,999), the model generates unreliable predictions because it is being asked to interpolate spaces it never saw during training. Robust systems must respect their training distribution boundaries.

---

## 4. System Architecture

The architecture enforces a strict separation of concerns:

```text
[ Frontend UI ] <--- JSON via REST ---> [ FastAPI Backend ] ---> [ Scikit-Learn Pipeline ]
     (HTML/JS)                             (Pydantic/Uvicorn)       (Model + Preprocessor)
```

1. **Frontend**: A minimal, vanilla JS/HTML layer responsible purely for UX, payload construction, and input boundary validation (preventing egregious OOD inputs).
2. **FastAPI**: The backend utilizes a `lifespan` hook. The pickled model (`model_pipeline.pkl`) is loaded **only once at startup into memory**, ensuring millisecond-level `predict` latency. 
3. **Inference Flow**: Incoming requests are validated against Pydantic schemas, transformed into Pandas DataFrames (matching the training pipeline schema exactly), passed through the model via `predict_proba`, thresholded, and returned as a serialized JSON probability.

---

## 5. API Documentation

### `GET /`
Health check endpoint.
**Response**: `{"status": "ok", "message": "Fraud Detection API running"}`

### `POST /predict`
Processes a transaction array and returns the fraud classification.

**Request Schema (JSON)**:
Requires `Time`, `V1`-`V28` (floats), and `Amount` (float).
```json
{
  "Time": 406.0,
  "V1": -2.31222,
  "V2": 1.95199,
   ...
  "Amount": 150.00
}
```

**Response**:
```json
{
  "fraud_probability": 0.999879,
  "prediction": 1,
  "threshold_used": 0.3899
}
```

---

## 6. User Workflow

The frontend is designed for interactive demonstration and debugging:
1. **Scenario Selection**: The user selects a "Legitimate" or "Fraudulent" predefined profile. This auto-loads the complex V1-V28 PCA vectors derived from real dataset examples behind the scenes.
2. **Amount Modification**: The user can manually tweak the transaction `Amount` to see how value changes affect the probability output.
3. **Submission**: The JS client validates the bounds of the `Amount` and shoots a `POST` request to FastAPI.
4. **Result Representation**: The UI visually parses the `fraud_probability`, updating a gauge UI and displaying "FRAUD DETECTED" or "LEGITIMATE" based on the backend's explicit threshold result.

---

## 7. Example Inputs

### Fraud Example (REAL DATA)
This is an exact, high-confidence fraud capture from the training set.
```json
{
    "Time": 406,
    "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
    "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657,
    "V9": -2.770089, "V10": -2.772272, "V11": 3.202033, "V12": -2.899907,
    "V13": -0.595222, "V14": -4.289254, "V15": 0.389724, "V16": -1.140747,
    "V17": -2.830056, "V18": -0.016822, "V19": 0.416956, "V20": 0.126911,
    "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
    "V25": 0.044519, "V26": 0.177840, "V27": 0.261145, "V28": -0.143276,
    "Amount": 100.00
}
```

### Legit Example (REALISTIC)
A standard, unsuspicious profile representing normal credit card utilization.
```json
{
    "Time": 0.0,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
    "Amount": 149.62
}
```

---

## 8. Key Challenges & Learnings

1. **Why Accuracy is a Lie**: Early modeling iterations showed 99.8% accuracy, which seemed perfect until confusion matrices revealed the model caught *zero* fraud cases. This reinforced the necessity of precision-recall tracking.
2. **PR-AUC vs ROC-AUC**: Switching our primary validation metric from ROC-AUC to PR-AUC forced the model tuning phase to care exclusively about how well we separate the microscopic fraction of minority class data, rather than measuring how well we predict the abundant majority class.
3. **The Importance of Thresholds**: Model confidence isn't binary. Shifting the threshold from 0.5 to 0.39 sacrificed ~1% precision to gain ~15% recall. In banking, analyzing a few more flagged accounts is vastly preferable to clearing a fraudulent $10k charge.
4. **Out of Distribution Sensitivities**: Inputting `Amount: 900,000` causes modern tree-based models to hallucinate because tree logic relies on bounding splits learned during training. We countered this by enforcing realistic input constraints at the API and UI boundaries.

---

## 9. How to Run

### Setup Environment
```bash
conda env create -f environment.yml
conda activate fraud-detection
```

### Start the Backend (FastAPI)
Run from the root `credit-card-fraud-v1` directory:
```bash
uvicorn src.api.main:app --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

### Start the Frontend
Simply spin up a local server inside the `frontend` folder:
```bash
cd frontend
python -m http.server 8001
```
Navigate to `http://localhost:8001` to use the UI.

---

## 10. Future Improvements

* **OOD Detection Layer**: Integrate an Isolation Forest or Autoencoder preprocessing step to explicitly reject payloads that fall outside the mathematical bounds of the training distribution before they hit the classifier.
* **Model Monitoring**: Integrate tracking hooks (e.g., evidently AI) to measure prediction drift over time.
* **Deployment**: Dockerize the application and deploy the API via Render/Railway, with the static frontend hosted on Netlify.

---

## 11. Tech Stack

* **ML / Data Science**: Python, Scikit-learn, Pandas, Numpy, Imbalanced-learn (SMOTE)
* **Backend API**: FastAPI, Uvicorn, Pydantic
* **Frontend**: Vanilla HTML5, CSS3 (Custom Variables/Properties), Modern JavaScript (Fetch API)
* **Model Serialization**: Joblib
