import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
# ----------------------------
# 1. Load CSV
# ----------------------------
df = pd.read_csv("data/train.csv")
df.columns = df.columns.str.strip()
# ----------------------------
# 2. Load tokenizer and model
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

gru_hidden_dim = 256
gru = nn.GRU(
    input_size=768,
    hidden_size=gru_hidden_dim,
    num_layers=1,
    batch_first=True,
    bidirectional=False
)

regressor = nn.Linear(gru_hidden_dim, 1)

# ----------------------------
# 3. Inference loop (NO TRAINING)
# ----------------------------
bert.eval()
gru.eval()
regressor.eval()

predicted_scores = []

with torch.no_grad():
    for idx, row in df.iterrows():
        context = row["context"]
        question = row["question"]
        answer = row["student_answer"]

        # Construct input text
        if isinstance(context, str) and context.strip() != "":
            text = f"context: {context}\nquestion: {question}\nanswer: {answer}"
        else:
            text = f"question: {question }\nanswer: {answer}"

        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length
        )

        # DistilBERT forward
        bert_out = bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
        last_hidden_state = bert_out.last_hidden_state

        # GRU forward
        gru_out, h_n = gru(last_hidden_state)

        # Regression
        score_pred = regressor(h_n.squeeze(0))
        predicted_scores.append(score_pred.item())

# ----------------------------
# 4. Attach predictions
# ----------------------------
df["predicted_score"] = predicted_scores
print(df[["question", "student_answer", "predicted_score"]].head())
