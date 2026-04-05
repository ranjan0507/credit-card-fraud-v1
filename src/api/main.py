"""
Main FastAPI app for serving the ML model.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from contextlib import asynccontextmanager

from src.api.schemas import Transaction
from src.api.model_loader import load_model_from_disk, get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    try:
        load_model_from_disk()
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield

app = FastAPI(title="Credit Card Fraud API" , lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Fraud Detection API running"
    }

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        model, threshold = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
        
    try:
        # 1. Convert input to pandas DataFrame (matching training column order)
        # Note: 'Class' doesn't exist at inference.
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict], columns=cols)
        
        # 2. Call model.predict_proba()
        proba = model.predict_proba(df)
        
        # 3. Extract fraud probability (class 1)
        fraud_probability = float(proba[0][1])
        
        # 4. Apply threshold
        prediction = 1 if fraud_probability >= threshold else 0
        
        return {
            "fraud_probability": fraud_probability,
            "prediction": prediction,
            "threshold_used": threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
