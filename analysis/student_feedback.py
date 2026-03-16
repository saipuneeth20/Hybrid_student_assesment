import pandas as pd
import torch
from transformers import DistilBertTokenizerFast

from src.models.student_scorer import StudentScorer
from src.utils.config import (
    BERT_MODEL_NAME,
    CHECKPOINT_PATH
)

# -------------------------
# Configuration
# -------------------------
STUDENT_FILE = "data/demo_single_student.csv"

STRONG_THRESHOLD = 75
MODERATE_THRESHOLD = 50

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load model
# -------------------------
model = StudentScorer(
    bert_model_name=BERT_MODEL_NAME
).to(device)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
)
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

# -------------------------
# Load student data
# -------------------------
df = pd.read_csv(STUDENT_FILE)
df.columns = df.columns.str.strip().str.lower()

df["answer"] = df["answer"].fillna("")
df["context"] = df["context"].fillna("") if "context" in df.columns else ""

# -------------------------
# Inference
# -------------------------
predictions = []
lengths = []

with torch.no_grad():
    for _, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        context = row["context"]

        text = (
            f"context: {context}\nquestion: {question}\nanswer: {answer}"
            if context.strip()
            else f"question: {question}\nanswer: {answer}"
        )

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=tokenizer.model_max_length
        )

        pred = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device)
        ).item() * 100.0

        predictions.append(pred)
        lengths.append(len(answer.split()))

df["predicted_score"] = predictions
df["answer_length"] = lengths

# -------------------------
# Strength classification
# -------------------------
def classify(score):
    if score >= STRONG_THRESHOLD:
        return "Strong"
    elif score >= MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Weak"

df["assessment"] = df["predicted_score"].apply(classify)

# -------------------------
# Feedback generation
# -------------------------
def generate_feedback(row):
    score = row["predicted_score"]
    length = row["answer_length"]

    if score >= STRONG_THRESHOLD:
        return "Answer is well-structured and conceptually clear."
    elif score >= MODERATE_THRESHOLD:
        if length < 5:
            return "Answer shows partial understanding but lacks detail."
        else:
            return "Answer is mostly correct but could be more precise."
    else:
        if length < 3:
            return "Answer is too brief and lacks explanation."
        else:
            return "Answer does not clearly address the question."

df["feedback"] = df.apply(generate_feedback, axis=1)

# -------------------------
# Student summary
# -------------------------
strong_count = (df["assessment"] == "Strong").sum()
weak_count = (df["assessment"] == "Weak").sum()

print("\nSTUDENT PERFORMANCE REPORT")
print("=" * 40)

for _, row in df.iterrows():
    print(f"\nQuestion: {row['question']}")
    print(f"Predicted Score: {row['predicted_score']:.1f}")
    print(f"Assessment: {row['assessment']}")
    print(f"Feedback: {row['feedback']}")

print("\nOVERALL SUMMARY")
print("-" * 40)
print(f"Strong answers: {strong_count}")
print(f"Weak answers:   {weak_count}")

if weak_count > strong_count:
    print("Overall: Student needs improvement in answer completeness.")
else:
    print("Overall: Student shows generally good understanding.")

