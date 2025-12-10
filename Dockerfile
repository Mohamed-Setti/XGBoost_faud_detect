FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .  .

ENV MODEL_PATH=/app/xgb_fraud_model_no_smote.pkl
ENV FEATURES_PATH=/app/feature_columns.npy
ENV PREPROCESSOR_PATH=/app/preprocess_pipeline.pkl
ENV META_PATH=/app/model_meta.json

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]