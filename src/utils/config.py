# src/utils/config.py

# =========================
# Model Configuration
# =========================

BERT_MODEL_NAME = "distilbert-base-uncased"

GRU_HIDDEN_SIZE = 256
GRU_NUM_LAYERS = 1
GRU_BIDIRECTIONAL = False


# =========================
# Tokenization Configuration
# =========================

MAX_SEQ_LENGTH = 384
TRUNCATION = True


# =========================
# Training Configuration
# =========================

LEARNING_RATE = 5e-5
EPOCHS = 10

LOSS_FUNCTION = "smooth_l1"   # descriptive only


# =========================
# Paths
# =========================
TRAIN_DATA_PATH = "data/train_combined.csv"
DEMO_DATA_PATH = "data/demo_single_student.csv"
CHECKPOINT_PATH = "checkpoints/student_scorer.pt"
TRAINING_LOG_PATH = "logs/training_log.csv"
# =========================
# Reproducibility
# =========================

RANDOM_SEED = 42