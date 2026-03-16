# app/main.py

from fastapi import FastAPI
from app.api_inference import run_inference
from app.classify import classify_score
from app.schemas import EvaluationRequest
from app.schemas import EvaluationResponse

app = FastAPI(
    title="Hybrid DistilBERT–GRU Student Answer Scoring API",
    version="1.0"
)


@app.get("/health")
def health():
    return {"status": "Model Loaded Successfully"}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest):
    score = run_inference(payload.question, payload.answer)
    label = classify_score(score)

    return {
        "score": round(score, 2),
        "classification": label
    }
