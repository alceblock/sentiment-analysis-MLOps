from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import os

from contextlib import asynccontextmanager
from typing import Optional
from collections import deque, Counter as PyCounter

from model_app.model_utility import MODEL_PATH, LATEST_MODEL_PATH, save_default_model

from monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SENTIMENT_COUNT,
    MODEL_ACCURACY,
    MODEL_PRECISION,
    MODEL_RECALL,
    MODEL_F1,
    CORRECT_PREDICTIONS,
    TOTAL_LABELED,
    PREDICTION_CONFIDENCE,
    INPUT_TEXT_LENGTH
)

# model_inference.py is called to run by Dockerfile once the app is started
# this function is used to save the default model if no other version already exists.
@asynccontextmanager
async def lifespan(_: FastAPI):
    if not (os.path.exists(LATEST_MODEL_PATH) and os.path.exists(MODEL_PATH)):
        print("Saving pre-defined model...")
        save_default_model()
    yield

app = FastAPI(lifespan=lifespan)

# Prometheus Endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Load model
sentiment_task = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)


# sliding window to check accuracy real-time
window = deque(maxlen=100)

# confusion matrix
conf_matrix = PyCounter()

# input
class TextInput(BaseModel):
    text: str
    label: Optional[str] = None

# inference api
@app.post("/predict")
async def predict(data: TextInput):

    # request metrics
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    start = time.time()

    # input monitoring
    INPUT_TEXT_LENGTH.observe(len(data.text))

    # inference
    result = sentiment_task(data.text)[0]

    # confidence monitoring
    PREDICTION_CONFIDENCE.observe(result["score"])

    label_pred = result["label"].lower()

    # sentiment distribution
    SENTIMENT_COUNT.labels(label=label_pred).inc()

    # latency
    latency = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

    # real-time performance
    if data.label:
        true_label = data.label.lower()

        TOTAL_LABELED.inc()

        correct = int(true_label == label_pred)

        if correct:
            CORRECT_PREDICTIONS.inc()

        # sliding window accuracy
        window.append(correct)
        accuracy = sum(window) / len(window)
        MODEL_ACCURACY.set(accuracy)

        # update confusion matrix
        conf_matrix[(true_label, label_pred)] += 1

        # precision/recall/F1
        labels = ["positive", "negative", "neutral"]

        precision_list = []
        recall_list = []

        for lbl in labels:
            tp = conf_matrix[(lbl, lbl)]
            fp = sum(conf_matrix[(other, lbl)] for other in labels if other != lbl)
            fn = sum(conf_matrix[(lbl, other)] for other in labels if other != lbl)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)

        precision_macro = sum(precision_list) / len(labels)
        recall_macro = sum(recall_list) / len(labels)

        f1_macro = (
            2 * precision_macro * recall_macro / (precision_macro + recall_macro)
            if (precision_macro + recall_macro) > 0 else 0
        )

        MODEL_PRECISION.set(precision_macro)
        MODEL_RECALL.set(recall_macro)
        MODEL_F1.set(f1_macro)

    return {
        "label": result["label"],
        "confidence": result["score"],
        "user_provided_label": data.label
    }

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)