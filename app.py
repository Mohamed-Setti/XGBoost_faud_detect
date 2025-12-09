import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict_utils import (
    load_features,
    load_metadata,
    load_pickle_or_joblib,
    read_csv_bytes,
    run_prediction_pipeline,
    get_preprocessor_expected_features,
)

# Config via environment variables (or defaults) 

#V1
MODEL_PATH = os.getenv("MODEL_PATH", "xgb_fraud_model_no_smote.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_columns.npy")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "preprocess_pipeline.pkl")
META_PATH = os.getenv("META_PATH", "model_meta.json")


#V2
# MODEL_PATH = os.getenv("MODEL_PATH", "xgb_fraud_detect_model_v2.pkl")
# FEATURES_PATH = os.getenv("FEATURES_PATH", "FEATURES_NPY.npy")
# PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "PREPROCESS_PKL.pkl")
# META_PATH = os.getenv("META_PATH", "META_JSON.json")

app = FastAPI(title="Model Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_artifacts():
    try:
        app.state.model = load_pickle_or_joblib(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

    try:
        app.state.features = load_features(FEATURES_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load features from {FEATURES_PATH}: {e}")

    if os.path.exists(PREPROCESSOR_PATH):
        try:
            app.state.preprocessor = load_pickle_or_joblib(PREPROCESSOR_PATH)
        except Exception:
            app.state.preprocessor = None
    else:
        app.state.preprocessor = None

    app.state.metadata = load_metadata(META_PATH) if os.path.exists(META_PATH) else {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    """
    Return the model feature list and inferred / metadata-provided types.
    Also includes preprocessor expected features if available.
    """
    features = getattr(app.state, "features", None)
    if features is None:
        raise HTTPException(status_code=500, detail="Features not loaded")
    metadata = getattr(app.state, "metadata", {}) or {}

    types_map = {}
    meta_cat = set(metadata.get("categorical_columns", []) or metadata.get("categorical", []) or [])
    meta_num = set(metadata.get("numeric_columns", []) or metadata.get("numeric", []) or [])

    for f in features:
        if f in meta_cat:
            types_map[f] = "categorical"
        elif f in meta_num:
            types_map[f] = "numeric"
        else:
            if f.lower() == "type" or any(k in f.lower() for k in ["cat", "category", "method", "channel"]):
                types_map[f] = "categorical"
            else:
                types_map[f] = "numeric"

    categorical_values = metadata.get("feature_values", {}) or metadata.get("categories", {}) or {}

    # If preprocessor can declare expected input feature names, include them
    preproc_expected = None
    if getattr(app.state, "preprocessor", None) is not None:
        expected = get_preprocessor_expected_features(app.state.preprocessor)
        if expected:
            preproc_expected = expected

    return {
        "features": features,
        "types": types_map,
        "categorical_values": categorical_values,
        "preprocessor_expected_features": preproc_expected,
    }


@app.post("/predict")
async def predict(file: Optional[UploadFile] = File(None), json_rows: Optional[str] = Form(None)):
    if file is None and json_rows is None:
        raise HTTPException(status_code=400, detail="Provide file (CSV) or json_rows form field")

    try:
        if file:
            content = await file.read()
            df = read_csv_bytes(content)
        else:
            import json

            rows = json.loads(json_rows)
            if not isinstance(rows, list):
                raise ValueError("json_rows must be a JSON array of row objects")
            import pandas as pd

            df = pd.DataFrame(rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse input data: {e}")

    try:
        result = run_prediction_pipeline(
            model=app.state.model,
            feature_columns=app.state.features,
            preprocessor=app.state.preprocessor,
            metadata=app.state.metadata,
            df=df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)