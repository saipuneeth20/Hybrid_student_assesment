import pandas as pd
import torch
from transformers import DistilBertTokenizerFast

from src.models.student_scorer import StudentScorer
from src.utils.config import BERT_MODEL_NAME, CHECKPOINT_PATH

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "data/demo_single_student.csv"

THRESHOLDS = {
    "easy":   {"strong": 70, "moderate": 50},
    "medium": {"strong": 75, "moderate": 55},
    "hard":   {"strong": 60, "moderate": 40},
}

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD MODEL
# =====================================================
model = StudentScorer(bert_model_name=BERT_MODEL_NAME).to(device)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
)
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# -------------------------
# SAFE COLUMN HANDLING
# -------------------------

# Student ID (CRITICAL FIX)
if "student" in df.columns:
    df["student"] = df["student"].astype(str)
else:
    # Assume single student file
    df["student"] = "STUDENT_001"

# Mandatory text fields
df["answer"] = df["answer"].fillna("")
df["context"] = df["context"].fillna("")
df["subject"] = df["subject"].fillna("unknown")
df["topic"] = df["topic"].fillna("unknown")

# Difficulty (SAFE)
if "difficulty" in df.columns:
    df["difficulty"] = (
        df["difficulty"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("nan", "")
    )
    df["difficulty"] = df["difficulty"].apply(
        lambda x: x if x in ["easy", "medium", "hard"] else "medium"
    )
else:
    df["difficulty"] = "medium"

# =====================================================
# MODEL INFERENCE
# =====================================================
predicted_scores = []

with torch.no_grad():
    for _, row in df.iterrows():
        text = (
            f"context: {row['context']}\n"
            f"question: {row['question']}\n"
            f"answer: {row['answer']}"
            if row["context"].strip()
            else f"question: {row['question']}\nanswer: {row['answer']}"
        )

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length
        )

        score = model(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device)
        ).item() * 100.0

        predicted_scores.append(score)

df["predicted_score"] = predicted_scores

# =====================================================
# DIFFICULTY-AWARE CLASSIFICATION
# =====================================================
def classify(row):
    rules = THRESHOLDS[row["difficulty"]]
    score = row["predicted_score"]

    if score >= rules["strong"]:
        return "Strong"
    elif score >= rules["moderate"]:
        return "Moderate"
    else:
        return "Weak"

df["assessment"] = df.apply(classify, axis=1)

# =====================================================
# STUDENT-WISE REPORT
# =====================================================
for student_id, sdf in df.groupby("student"):

    print("\n" + "=" * 70)
    print(f"STUDENT REPORT — ID: {student_id}")
    print("=" * 70)

    # Per-question output
    for _, r in sdf.iterrows():
        print(
            f"[{r['assessment']}] "
            f"Subject: {r['subject']} | "
            f"Topic: {r['topic']} | "
            f"Difficulty: {r['difficulty']} | "
            f"Score: {r['predicted_score']:.1f}"
        )

    # Aggregation
    strong_topics = (
        sdf[sdf["assessment"] == "Strong"]
        .groupby("topic")
        .size()
        .sort_values(ascending=False)
    )

    weak_topics = (
        sdf[sdf["assessment"] == "Weak"]
        .groupby("topic")
        .size()
        .sort_values(ascending=False)
    )

    print("\nSTRENGTHS:")
    if len(strong_topics):
        for t in strong_topics.index:
            print(f"✓ {t}")
    else:
        print("No strong areas identified.")

    print("\nWEAKNESSES:")
    if len(weak_topics):
        for t in weak_topics.index:
            print(f"✗ {t}")
    else:
        print("No critical weaknesses detected.")

    print("\nRECOMMENDATION:")
    if len(weak_topics):
        print(
            "Focus on improving conceptual clarity and explanation depth, "
            "especially in easier and medium difficulty questions."
        )
    else:
        print(
            "Student demonstrates strong understanding. Encourage more hard questions."
        )
