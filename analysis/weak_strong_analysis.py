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
# Device (inference-safe)
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
# Load dataset
# -------------------------
df = pd.read_csv(TRAIN_DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Safety
df["answer"] = df["answer"].fillna("")
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

# -------------------------
# Inference
# -------------------------
predictions = []

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
        ).item()* 100.0

        predictions.append(pred)

df["predicted_score"] = predictions

# -------------------------
# Weak / Strong split
# -------------------------
low_q = df["predicted_score"].quantile(0.25)
high_q = df["predicted_score"].quantile(0.75)

weak_df = df[df["predicted_score"] <= low_q]
strong_df = df[df["predicted_score"] >= high_q]

# -------------------------
# Helper metrics
# -------------------------
def summarize(group, name):
    mae = (group["predicted_score"] - group["score"]).abs().mean()
    avg_len = group["answer"].str.split().str.len().mean()

    print(f"\n{name} GROUP")
    print("-" * 30)
    print(f"Samples: {len(group)}")
    print(f"Mean predicted score: {group['predicted_score'].mean():.2f}")
    print(f"Mean true score:      {group['score'].mean():.2f}")
    print(f"Mean absolute error:  {mae:.2f}")
    print(f"Avg answer length:    {avg_len:.1f} words")

# -------------------------
# Print summaries
# -------------------------
summarize(weak_df, "WEAK")
summarize(strong_df, "STRONG")

# -------------------------
# Representative examples
# -------------------------
print("\nRepresentative WEAK examples:")
print(weak_df[["question", "answer", "score", "predicted_score"]].head(3))

print("\nRepresentative STRONG examples:")
print(strong_df[["question", "answer", "score", "predicted_score"]].head(3))
