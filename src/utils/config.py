# src/utils/config.py

BERT_MODEL_NAME = "distilbert-base-uncased"

GRU_HIDDEN_SIZE = 256
GRU_NUM_LAYERS = 1
GRU_BIDIRECTIONAL = False

MAX_SEQ_LENGTH = 384
TRUNCATION = True

LEARNING_RATE = 5e-5
EPOCHS = 30                              # increased from 10

TRAIN_DATA_PATH = "data/train_clean.csv" # fixed — was train_combined.csv
DEMO_DATA_PATH = "data/demo_single_student.csv"
CHECKPOINT_PATH = "checkpoints/student_scorer_v3.pt"  # new checkpoint name
TRAINING_LOG_PATH = "logs/training_log_v3.csv"

RANDOM_SEED = 42