import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
# ============================
# 1. Load and clean CSV
# ============================
df = pd.read_csv("data/train.csv")

# normalize column names
df.columns = df.columns.str.strip().str.lower()

# make sure score is numeric (keeps 0, drops invalid)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

# ============================
# 2. Load tokenizer & models
# ============================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)
bert = DistilBertModel.from_pretrained(
    "distilbert-base-uncased"
)
gru_hidden_dim = 256
gru = nn.GRU(
    input_size=768,
    hidden_size=gru_hidden_dim,
    num_layers=1,
    batch_first=True,
    bidirectional=False
)
regressor = nn.Linear(gru_hidden_dim, 1)
# ============================
# 3. Loss & optimizer
# ============================
criterion = nn.SmoothL1Loss()  # Huber loss
optimizer = torch.optim.AdamW(
    list(bert.parameters()) +
    list(gru.parameters()) +
    list(regressor.parameters()),
    lr=2e-5
)
# ============================
# 4. Training loop
# ============================
bert.train()
gru.train()
regressor.train()
EPOCHS = 10  # keep 1 for now
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}\n" + "-" * 30)

    for idx, row in df.iterrows():

        # ----- extract fields safely -----
        question = row["question"]

        context = row["context"] if isinstance(row["context"], str) else ""
        answer = row["answer"] if isinstance(row["answer"], str) else ""

        true_score = torch.tensor(
            [float(row["score"])],
            dtype=torch.float
        )

        # ----- construct input text -----
        if context.strip() != "":
            text = (
                f"context: {context}\n"
                f"question: {question}\n"
                f"answer: {answer}"
            )
        else:
            text = (
                f"question: {question}\n"
                f"answer: {answer}"
            )

        # ----- tokenize -----
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=tokenizer.model_max_length
        )

        # ----- forward pass -----
        bert_out = bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

        _, h_n = gru(bert_out.last_hidden_state)

        pred_score = regressor(
            h_n.squeeze(0)
        ).squeeze(1)   # shape (1)

        # ----- loss -----
        loss = criterion(pred_score, true_score)

        # ----- backprop -----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Row {idx} | "
            f"True: {true_score.item():.1f} | "
            f"Pred: {pred_score.item():.2f} | "
            f"Loss: {loss.item():.2f}"
        )