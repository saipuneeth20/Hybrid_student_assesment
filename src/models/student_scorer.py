import torch
import torch.nn as nn
from transformers import DistilBertModel


class StudentScorer(nn.Module):
    """
    Hybrid DistilBERT + GRU model for student answer scoring (regression).

    Architecture:
        DistilBERT → GRU → Dropout → Linear(1)
    """

    def __init__(
        self,
        bert_model_name: str = "distilbert-base-uncased",
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
    ):
        super(StudentScorer, self).__init__()

        # -----------------------------
        # DistilBERT Backbone
        # -----------------------------
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768

        # -----------------------------
        # GRU Layer
        # -----------------------------
        self.gru = nn.GRU(
            input_size=bert_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # -----------------------------
        # Dropout (for variance + regularization)
        # -----------------------------
        self.dropout = nn.Dropout(0.3)

        # -----------------------------
        # Regression Head
        # -----------------------------
        self.regressor = nn.Linear(gru_hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # -----------------------------
        # 1️⃣ DistilBERT Forward
        # -----------------------------
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Token embeddings
        token_embeddings = bert_outputs.last_hidden_state
        # Shape: (batch_size, seq_len, 768)

        # -----------------------------
        # 2️⃣ GRU Forward
        # -----------------------------
        _, hidden_state = self.gru(token_embeddings)
        # hidden_state shape: (num_layers, batch_size, hidden_size)

        # Take last layer hidden state
        final_hidden = hidden_state[-1]
        # Shape: (batch_size, hidden_size)

        # -----------------------------
        # 3️⃣ Dropout
        # -----------------------------
        final_hidden = self.dropout(final_hidden)

        # -----------------------------
        # 4️⃣ Regression Output
        # -----------------------------
        score = self.regressor(final_hidden)
        # Shape: (batch_size, 1)

        return score