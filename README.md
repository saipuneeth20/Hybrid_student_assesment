# System Architecture

The system follows a multi-stage deep learning pipeline:

```text
Question + Reference Answer + Student Answer
                    ↓
          Input Concatenation Layer
                    ↓
         DistilBERT Semantic Encoder
                    ↓
         Contextual Token Embeddings
                    ↓
           GRU Sequential Modeling
                    ↓
         Dropout + Linear Regression
                    ↓
            Sigmoid Score Output
                    ↓
             Final Score (0–100)
```

---

# Why Hybrid DistilBERT + GRU?

## DistilBERT

DistilBERT provides:

- Contextual semantic embeddings
- Lightweight transformer architecture
- Reduced parameter count compared to BERT
- Faster inference and lower memory usage

### DistilBERT Specifications

| Parameter | Value |
|---|---|
| Parameters | 66 Million |
| Hidden Size | 768 |
| Transformer Layers | 6 |
| Attention Heads | 12 |
| Tokenizer | WordPiece |
| Base Model | distilbert-base-uncased |

---

## GRU (Gated Recurrent Unit)

GRU provides:

- Sequential dependency modeling
- Positional understanding
- Lightweight recurrent computation
- Faster training than LSTM

The GRU processes the sequence of DistilBERT embeddings and captures:

- Logical flow
- Explanation order
- Structural coherence
- Sequential patterns

### GRU Specifications

| Parameter | Value |
|---|---|
| Hidden Size | 256 |
| Layers | 1 |
| Bidirectional | False |
| Output Dimension | 256 |

---

# Input Representation

The model uses a three-part input structure:

```text
question: {QUESTION}
reference: {REFERENCE_ANSWER}
student: {STUDENT_ANSWER}
```

This enables cross-attention between:

- Question
- Reference Answer
- Student Answer

allowing the model to perform semantic comparison-based evaluation.

---

# Dataset Information

## Dataset Statistics

| Parameter | Value |
|---|---|
| Total Samples | 440 |
| Original Samples | 284 |
| Augmented Samples | 156 |
| Score Range | 2–100 |
| Subjects | English, Mathematics, Science, ICT |

---

## Augmentation Strategy

The dataset includes synthetic contrastive augmentation covering:

| Score Range | Description |
|---|---|
| 95–98 | Perfect matches |
| 85–92 | Correct paraphrases |
| 60–80 | Partially correct answers |
| 30–55 | Vague but relevant |
| 10–25 | Incorrect but topic-related |
| 2–8 | Completely irrelevant |

---

# Data Preprocessing Pipeline

The preprocessing pipeline includes:

1. Text Cleaning
2. Lowercase normalization
3. Tokenization using DistilBertTokenizerFast
4. Sequence padding and truncation
5. Label normalization
6. Input concatenation

---

# Training Configuration

| Parameter | Value |
|---|---|
| Batch Size | 32 |
| Epochs | 30 |
| Best Epoch | 11 |
| Learning Rate (BERT) | 2e-5 |
| Learning Rate (GRU) | 5e-5 |
| Optimizer | AdamW |
| Loss Function | MSELoss |
| Dropout | 0.3 |
| Max Sequence Length | 384 |

---

# Evaluation Metrics

| Metric | Value |
|---|---|
| Validation Loss | 0.0140 |
| Validation MAE | 0.0869 |
| Approximate MAE (0–100 Scale) | ~8.7 Points |
| RMSE | ~11.8 Points |
| Best Epoch | 11 |

---

# Performance Highlights

The model successfully distinguishes:

| Response Type | Predicted Score |
|---|---|
| Correct Semantic Match | 77 |
| Vague but Relevant | 16 |
| Completely Irrelevant | 9 |
| Long Irrelevant Answer | 15 |

This demonstrates that the system evaluates semantic correctness rather than answer length.

---

# Project Structure

```text
Hybrid_student_assesment/
│
├── analysis/
│   ├── error_case_analysis.py
│   ├── student_feedback.py
│   └── weak_strong_analysis.py
│
├── app/
│   ├── api_inference.py
│   ├── classify.py
│   ├── main.py
│   └── schemas.py
│
├── scripts/
│   ├── combine_data.py
│   ├── generate_dataset.py
│   ├── humanizeanswers.py
│   ├── prune.py
│   └── rebuild_dataset.py
│
├── src/
│   ├── inference/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── verification/
│
├── checkpoints/
├── data/
├── logs/
├── README.md
└── requirements.txt
```

---

# Technologies Used

## Programming Language

- Python 3.11

## Machine Learning Frameworks

- PyTorch
- HuggingFace Transformers
- Scikit-learn
- NumPy
- Pandas

## Deployment

- FastAPI
- Uvicorn
- Pydantic

## Visualization and Utilities

- Matplotlib
- Git
- GitHub

---

# Installation

Clone the repository:

```bash
git clone https://github.com/saipuneeth20/Hybrid_student_assesment.git
cd Hybrid_student_assesment
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Model Training

Run training:

```bash
python src/training/train_v2.py
```

Model checkpoints are stored in:

```text
checkpoints/
```

Logs are stored in:

```text
logs/
```

---

# Inference

## Single Sample Inference

```bash
python src/inference/infer.py \
  --student_answer "Plants use sunlight to produce food." \
  --reference_answer "Photosynthesis is the process by which plants make food using sunlight." \
  --checkpoint checkpoints/student_scorer_v3.pt
```

---

# FastAPI Deployment

Start the API server:

```bash
cd app
uvicorn main:app --reload
```

API endpoint:

```text
POST /evaluate
```

### Example Request

```json
{
  "question": "What is photosynthesis?",
  "reference_answer": "Photosynthesis is the process by which plants make food using sunlight.",
  "student_answer": "Plants use solar energy to produce glucose."
}
```

### Example Response

```json
{
  "score": 77.0,
  "classification": "Moderate"
}
```

---

# Classification Mapping

| Score Range | Classification |
|---|---|
| 80–100 | Strong |
| 60–79 | Moderate |
| 40–59 | Developing |
| 0–39 | Weak |

---

# Hardware Requirements

## Minimum Requirements

- Dual-core CPU
- 4 GB RAM
- 50 GB Storage

## Recommended Requirements

- Intel i5 / Ryzen 5 or higher
- 8–16 GB RAM
- SSD Storage
- NVIDIA GPU (optional)


---

# Research Publication

Published in:

**IJERT – International Journal of Engineering Research & Technology**

- Volume: 15
- Issue: 02
- Month: February 2026
- Registration ID: IJERTV15IS020339

