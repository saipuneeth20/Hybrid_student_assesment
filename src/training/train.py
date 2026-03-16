# src/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import csv
import os

from src.models.student_scorer import StudentScorer
from src.data.dataset import StudentAnswerDataset
from src.data.collate import StudentCollator
from src.utils.config import (
    BERT_MODEL_NAME,
    LEARNING_RATE,
    EPOCHS,
    TRAIN_DATA_PATH,
    CHECKPOINT_PATH
)

# =========================
# Training configuration
# =========================
BATCH_SIZE = 32
VAL_SPLIT_RATIO = 0.1
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
LOG_PATH = "logs/training_log.csv"
def compute_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels)).item()
def main():
    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # Dataset + split
    # =========================
    full_dataset = StudentAnswerDataset(TRAIN_DATA_PATH)
    dataset_size = len(full_dataset)

    val_size = int(dataset_size * VAL_SPLIT_RATIO)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    print(f"Total samples: {dataset_size}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # =========================
    # DataLoaders
    # =========================
    collator = StudentCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    # =========================
    # Model
    # =========================
    model = StudentScorer(
        bert_model_name=BERT_MODEL_NAME
    ).to(device)
    # =========================
    # Loss & Optimizer
    # =========================
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # =========================
    # Early stopping setup
    # =========================
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # =========================
    # Prepare log file
    # =========================
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mae"])

    # =========================
    # Training loop
    # =========================
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}")
        print("-" * 40)

        # -------- TRAIN --------
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = (batch["labels"] / 100.0).to(device)

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).squeeze(1)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps

        # -------- VALIDATION --------
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = (batch["labels"] / 100.0).to(device)
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).squeeze(1)

                loss = criterion(preds, labels)
                mae = compute_mae(preds, labels)
                val_loss_sum += loss.item()
                val_mae_sum += mae
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps
        avg_val_mae = val_mae_sum / val_steps
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f}")
        print(f"Val   MAE : {avg_val_mae:.4f}")
        # -------- LOG --------
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_mae"])
        # -------- EARLY STOPPING --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("✔ Best model saved")

        else:
            epochs_without_improvement += 1
            print(f"No improvement ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break

    print("\nTraining complete")


if __name__ == "__main__":
    main()
