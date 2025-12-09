# Small snippet showing the changes to include in your run_prediction_pipeline (or replace function body)
import numpy as np

# after preds, proba = predict_with_model(...)
out = {"predictions": preds.tolist()}
if proba is not None:
    out["proba"] = proba.tolist() if isinstance(proba, np.ndarray) else proba

# If the model exposes classes_ (sklearn classifiers), map ints -> labels
classes = getattr(model, "classes_", None)
if classes is not None:
    # classes could be [0,1] or strings; create a list of string labels in model order
    class_labels = [str(c) for c in classes]
    out["class_labels"] = class_labels
    # predicted_label for first sample (if preds are integers indexing classes)
    # If preds are already class values (not indices), just return them stringified
    try:
        # If preds are indices into classes (0/1 matching classes), this maps properly
        out["predicted_labels"] = [class_labels[int(p)] if int(p) < len(class_labels) else str(p) for p in preds]
    except Exception:
        out["predicted_labels"] = [str(p) for p in preds]
else:
    # Optionally load labels from metadata if available
    # metadata might contain {"class_labels": ["legit","fraud"]}
    if metadata and metadata.get("class_labels"):
        out["class_labels"] = metadata["class_labels"]
        out["predicted_labels"] = [out["class_labels"][int(p)] for p in preds]
    else:
        out["predicted_labels"] = [str(p) for p in preds]

out["sample_inputs"] = df_prepared.head(10).to_dict(orient="records")