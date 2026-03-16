import pandas as pd
import torch
from transformers import DistilBertTokenizerFast

from src.models.student_scorer import StudentScorer
from src.utils.config import (
    BERT_MODEL_NAME,
    CHECKPOINT_PATH,
    TRAIN_DATA_PATH
)

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
    torch.load(CHECKPOINT_PATH, map_location=device)
)
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(TRAIN_DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

df["answer"] = df["answer"].fillna("")
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

# -------------------------
# Inference
# -------------------------
preds = []

with torch.no_grad():
    for _, row in df.iterrows():
        question = row["question"]
        context = row["context"] if isinstance(row["context"], str) else ""
        answer = row["answer"]

        if context.strip():
            text = f"context: {context}\nquestion: {question}\nanswer: {answer}"
        else:
            text = f"question: {question}\nanswer: {answer}"

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=tokenizer.model_max_length
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        pred = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).item() * 100.0

        preds.append(pred)

df["predicted_score"] = preds
df["error"] = df["predicted_score"] - df["score"]

# -------------------------
# Error thresholds
# -------------------------
ERROR_THRESHOLD = 20.0

over_scored = df[df["error"] >= ERROR_THRESHOLD]
under_scored = df[df["error"] <= -ERROR_THRESHOLD]

# -------------------------
# Summary
# -------------------------
print("\nERROR CASE SUMMARY")
print("-" * 40)
print(f"Total samples: {len(df)}")
print(f"Over-scored weak answers: {len(over_scored)}")
print(f"Under-scored strong answers: {len(under_scored)}")

# -------------------------
# Show examples
# -------------------------
def show_examples(group, title, n=5):
    print(f"\n{title}")
    print("-" * 40)
    cols = ["question", "answer", "score", "predicted_score", "error"]
    print(group[cols].head(n))

show_examples(
    over_scored.sort_values("error", ascending=False),
    "MOST OVER-SCORED ANSWERS"
)

show_examples(
    under_scored.sort_values("error"),
    "MOST UNDER-SCORED ANSWERS"
)