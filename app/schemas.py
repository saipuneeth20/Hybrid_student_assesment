from pydantic import BaseModel, Field

class EvaluationRequest(BaseModel):
    question: str = Field(
        example="What is photosynthesis?"
    )
    reference_answer: str = Field(
        example="Photosynthesis is the process by which plants make food using sunlight and carbon dioxide."
    )
    student_answer: str = Field(
        example="Plants use sunlight to make their own food through photosynthesis."
    )

class EvaluationResponse(BaseModel):
    score: float
    classification: str