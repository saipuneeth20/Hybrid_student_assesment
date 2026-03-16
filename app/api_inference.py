# app/api_inference.py

import torch
from transformers import DistilBertTokenizer
from src.models.student_scorer import StudentScorer

DEVICE = torch.device("cpu")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load model
model = StudentScorer("distilbert-base-uncased")
model.load_state_dict(
    torch.load("checkpoints/student_scorer_v2_777.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()
print("Loaded checkpoint: student_scorer_v2_777.pt")

def run_inference(question: str, answer: str):
    combined_text = question + " [SEP] " + answer

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        raw_output = model(input_ids=input_ids, attention_mask=attention_mask)

    # raw_output shape: (1,1)
    normalized_score = raw_output.squeeze().item()

    # Rescale to 0–100
    final_score = normalized_score * 100.0

    return float(final_score)