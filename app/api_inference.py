# app/api_inference.py

import torch
from transformers import DistilBertTokenizer
from src.models.student_scorer import StudentScorer
from src.utils.text_builder import build_input_text

DEVICE = torch.device("cpu")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = StudentScorer("distilbert-base-uncased")
model.load_state_dict(
    torch.load("checkpoints/student_scorer_v3.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()
print("Loaded checkpoint: student_scorer_v3.pt")


def run_inference(question: str, reference_answer: str, student_answer: str, context: str = ""):
    text = build_input_text(
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        context=context
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=384
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        raw_output = model(input_ids=input_ids, attention_mask=attention_mask)

    # sigmoid is already applied in model — output is 0–1
    normalized_score = raw_output.squeeze().item()
    final_score = normalized_score * 100.0
    return float(final_score)