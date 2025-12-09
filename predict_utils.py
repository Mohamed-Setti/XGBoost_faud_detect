import io
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


def load_pickle_or_joblib(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f)


def load_features(path: str) -> List[str]:
    arr = np.load(path, allow_pickle=True)
    try:
        lst = arr.tolist()
    except Exception:
        lst = list(arr)
    return [str(x) for x in lst]


def load_metadata(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_preprocessor_expected_features(preprocessor: Any) -> Optional[List[str]]:
    """
    Try to infer the feature names the preprocessor expects as input.
    - If the preprocessor has `feature_names_in_` (sklearn >=0.24), use it.
    - Otherwise return None (caller should fallback to provided feature list).
    """
    if preprocessor is None:
        return None
    expected = getattr(preprocessor, "feature_names_in_", None)
    if expected is not None:
        try:
            return [str(x) for x in expected.tolist()] if hasattr(expected, "tolist") else [str(x) for x in expected]
        except Exception:
            return [str(x) for x in expected]
    return None


def prepare_input_dataframe(df: pd.DataFrame, feature_columns: List[str], metadata: Optional[Dict] = None) -> pd.DataFrame:
    """
    Ensure df has all feature_columns, reordered. Fill missing columns with defaults.
    """
    meta_defaults = {}
    if metadata:
        meta_defaults = metadata.get("feature_defaults", {}) or metadata.get("defaults", {})

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            default = meta_defaults.get(col, pd.NA)
            df[col] = default

    # Keep only feature_columns and reorder
    df = df[list(feature_columns)].copy()

    # Fill missing values
    for col in df.columns:
        if df[col].isna().any():
            if col in meta_defaults:
                df[col] = df[col].fillna(meta_defaults[col])
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("")

    return df


def apply_preprocessor(preprocessor: Any, df: pd.DataFrame) -> Any:
    """
    Apply the preprocessor. If transform fails with a DataFrame, try numpy values.
    """
    try:
        return preprocessor.transform(df)
    except Exception as e:
        try:
            return preprocessor.transform(df.values)
        except Exception as e2:
            raise RuntimeError(f"Preprocessor.transform failed: {e}; fallback failed: {e2}")


def predict_with_model(model: Any, X: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run model predictions. Returns (preds, proba) where proba may be None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        preds = model.predict(X)
        return np.asarray(preds), np.asarray(proba)
    if hasattr(model, "predict"):
        preds = model.predict(X)
        return np.asarray(preds), None
    try:
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X)
            proba = model.predict(dmat)
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            preds = np.argmax(proba, axis=1)
            return preds, proba
    except Exception:
        pass
    raise RuntimeError("Model does not have a supported predict / predict_proba interface.")


def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _map_preds_to_labels(preds: np.ndarray, model: Any, metadata: Optional[Dict]) -> Tuple[List[str], List[str]]:
    """
    Returns (class_labels, predicted_labels) where:
      - class_labels is a list of class labels (strings) in model order when available
      - predicted_labels is a list of string labels for each prediction
    Uses model.classes_ if available, otherwise checks metadata.class_labels, otherwise falls back to stringified preds.
    """
    class_labels = None
    predicted_labels: List[str] = []

    # Try sklearn-style classes_
    classes_attr = getattr(model, "classes_", None)
    if classes_attr is not None:
        try:
            class_labels = [str(c) for c in (classes_attr.tolist() if hasattr(classes_attr, "tolist") else classes_attr)]
        except Exception:
            class_labels = [str(c) for c in classes_attr]

    # Fallback to metadata
    if class_labels is None and metadata:
        meta_labels = metadata.get("class_labels") or metadata.get("labels")
        if isinstance(meta_labels, (list, tuple)) and len(meta_labels) > 0:
            class_labels = [str(x) for x in meta_labels]

    # If we have class labels, map predictions to labels when possible
    if class_labels is not None:
        for p in preds:
            # p might be an index (0/1) or the actual class value (e.g. 0/1 or 'fraud')
            try:
                # numeric index mapping
                if isinstance(p, (int, np.integer)):
                    idx = int(p)
                    if 0 <= idx < len(class_labels):
                        predicted_labels.append(class_labels[idx])
                        continue
                # p could be equal to a class value
                sval = str(p)
                if sval in class_labels:
                    predicted_labels.append(sval)
                    continue
                # try converting to int index
                idx = int(p)
                if 0 <= idx < len(class_labels):
                    predicted_labels.append(class_labels[idx])
                    continue
            except Exception:
                pass
            # fallback to stringified prediction
            predicted_labels.append(str(p))
        return class_labels, predicted_labels

    # No class label info: just stringify preds
    predicted_labels = [str(p) for p in preds.tolist()]
    return [], predicted_labels


def run_prediction_pipeline(
    model: Any,
    feature_columns: List[str],
    preprocessor: Optional[Any],
    metadata: Optional[Dict],
    df: pd.DataFrame,
) -> Dict:
    """
    Prepares input and runs predictions.
    - If the preprocessor exposes its expected input feature names, prefer those over the provided feature_columns list.
    - Returns predictions, probabilities (if available), sample_inputs and human-friendly label info.
    """
    # Determine the exact features to provide to the preprocessor
    if preprocessor is not None:
        expected = get_preprocessor_expected_features(preprocessor)
        feature_columns_used = expected if expected else feature_columns
    else:
        feature_columns_used = feature_columns

    # Prepare DataFrame to exactly match expected columns
    df_prepared = prepare_input_dataframe(df, feature_columns_used, metadata)

    # Apply preprocessor if present
    X = apply_preprocessor(preprocessor, df_prepared) if preprocessor else df_prepared.values

    # Predict
    preds, proba = predict_with_model(model, X)

    out: Dict[str, Any] = {}
    out["predictions"] = preds.tolist()
    if proba is not None:
        out["proba"] = proba.tolist() if isinstance(proba, np.ndarray) else proba

    # Add class label information and predicted labels where possible
    class_labels, predicted_labels = _map_preds_to_labels(preds, model, metadata)
    if class_labels:
        out["class_labels"] = class_labels
    out["predicted_labels"] = predicted_labels

    # Convenience: include predicted_label (first) and predicted_probability (first) if single-row request
    if len(predicted_labels) == 1:
        out["predicted_label"] = predicted_labels[0]
    if proba is not None:
        try:
            # For each row, compute the probability corresponding to the predicted class (best-effort)
            prob_for_preds = []
            for i, p in enumerate(preds):
                # If proba is 2D and p is numeric index -> take proba[i, p]
                if isinstance(proba, np.ndarray) and proba.ndim == 2:
                    try:
                        idx = int(p)
                        if 0 <= idx < proba.shape[1]:
                            prob_for_preds.append(float(proba[i, idx]))
                            continue
                    except Exception:
                        pass
                    # fallback: take max probability
                    prob_for_preds.append(float(np.max(proba[i])))
                else:
                    # proba not 2D: treat as single value per sample
                    prob_for_preds.append(float(proba[i]) if hasattr(proba, "__len__") else float(proba))
            out["predicted_probability"] = prob_for_preds[0] if len(prob_for_preds) == 1 else prob_for_preds
        except Exception:
            # ignore probability mapping errors
            pass

    # Include a small sample of inputs (after preparation/reordering) for debugging
    out["sample_inputs"] = df_prepared.head(10).to_dict(orient="records")

    # Optionally include some model metadata for debugging (class names, model type)
    try:
        out["_model_info"] = {
            "has_classes_attr": getattr(model, "classes_", None) is not None,
            "model_type": type(model).__name__,
        }
    except Exception:
        pass

    return out