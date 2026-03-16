# src/inference/infer.py

import torch
import pandas as pd
from transformers import DistilBertTokenizerFast

from src.models.student_scorer import StudentScorer
from src.utils.text_builder import build_input_text
from src.utils.config import (
    BERT_MODEL_NAME,
    MAX_SEQ_LENGTH,
    CHECKPOINT_PATH,
    DEMO_DATA_PATH
)


def main():
    # -------------------------
    # Load tokenizer
    # -------------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        BERT_MODEL_NAME
    )

    # -------------------------
    # Load model
    # -------------------------
    model = StudentScorer(
        bert_model_name=BERT_MODEL_NAME
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )
    model.load_state_dict(state_dict)

    model.eval()

    # -------------------------
    # Load demo data
    # -------------------------
    df = pd.read_csv(DEMO_DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    predictions = []

    # -------------------------
    # Inference loop
    # -------------------------
    with torch.no_grad():
        for _, row in df.iterrows():
            question = row.get("question", "")
            context = row.get("context", "")
            answer = row.get("answer", "")

            text = build_input_text(
                question=question,
                context=context,
                answer=answer
            )

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=MAX_SEQ_LENGTH
            )

            score = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )

            predictions.append(score.item())

    # -------------------------
    # Attach predictions
    # -------------------------
    df["predicted_score"] = predictions

    print(
        df[["question", "answer", "predicted_score"]].head()
    )


if __name__ == "__main__":
    main()
