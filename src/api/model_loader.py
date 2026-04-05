"""
Model loader that loads the model only once at startup.
"""
import joblib
import os

_model = None
_threshold = None

def load_model_from_disk():
    global _model, _threshold
    # Assume script is run from project root or locate path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_root, "artifacts", "model_pipeline.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    data = joblib.load(model_path)
    _model = data["model"]
    _threshold = float(data["threshold"])
    print("Model loaded successfully.")

def get_model():
    if _model is None or _threshold is None:
        raise RuntimeError("Model has not been loaded yet.")
    return _model, _threshold
