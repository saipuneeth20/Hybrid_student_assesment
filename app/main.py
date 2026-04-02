from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.api_inference import run_inference
from app.classify import classify_score
from app.schemas import EvaluationRequest, EvaluationResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Hybrid Student Assessment System",
        version="1.0",
        description="""
## DistilBERT + GRU Automated Answer Scoring

This API scores student short answers against a reference answer using a hybrid deep learning model.

### How to use
1. Click **POST /evaluate** below
2. Click **Try it out**
3. Fill in the three fields and click **Execute**

### Score Classification
| Score | Classification |
|-------|---------------|
| 80–100 | Strong |
| 60–79 | Moderate |
| 40–59 | Developing |
| 0–39 | Weak |
        """,
        routes=app.routes,
    )
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/health", tags=["System"])
def health():
    return {"status": "Model Loaded Successfully"}


@app.post("/evaluate", tags=["Scoring"], response_model=EvaluationResponse,
    summary="Score a student answer",
    description="Provide the question, reference answer, and student answer to get a score from 0–100.")
def evaluate(payload: EvaluationRequest):
    score = run_inference(
        payload.question,
        payload.reference_answer,
        payload.student_answer
    )
    label = classify_score(score)
    return {"score": round(score, 2), "classification": label}