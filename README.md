# Hybrid Student Assessment — Offline Answer Scoring System

An automated short-answer grading system built on a hybrid **DistilBERT + GRU** architecture,
designed for low-resource and offline-first deployment in rural education environments.

---

## Architecture

The model follows a three-stage pipeline:

```
Student Answer + Reference Answer
        ↓
  [DistilBERT Encoder]
  Contextual token embeddings (768-dim)
        ↓
  [Single-layer GRU]
  Sequential dependency across token states
        ↓
  [Linear Projection Head]
  Scalar score → rescaled to 0–100
```

**DistilBERT** serves as the frozen (or fine-tuned) encoder, producing contextual token-level
embeddings from the concatenated student and reference answer pair. The encoder captures
semantic meaning without requiring the full computational overhead of BERT.

**GRU** processes the sequence of token embeddings output by DistilBERT, capturing
positional and sequential dependencies across the token states. This is the key architectural
decision: rather than pooling embeddings directly into a fixed vector (which discards token
order), the GRU treats the embedding sequence as a temporal signal and distills it into a
final hidden state that preserves sequential structure.

**Linear Head** projects the GRU's final hidden state to a single scalar, which is then
rescaled to the 0–100 range to produce the final score.

### Why this architecture?

Standard transformer-only pipelines assume reliable internet connectivity and significant
compute for inference. This system is designed for deployment on low-spec hardware in rural
schools where GPU availability and network access cannot be guaranteed. DistilBERT provides
a compact semantic encoder (~66M parameters vs BERT's ~110M), and the GRU adds minimal
overhead while preserving sequential structure that a simple mean-pool would discard.

---

## Project Structure

```
Hybrid_student_assesment/
├── analysis/               # Post-training analysis scripts
│   ├── error_case_analysis.py
│   ├── student_difficulty_feedback.py
│   ├── student_feedback.py
│   └── weak_strong_analysis.py
├── app/                    # FastAPI demo inference API
│   ├── api_inference.py
│   ├── classify.py
│   ├── main.py
│   └── schemas.py
├── scripts/                # Dataset generation and preprocessing utilities
│   ├── combine_data.py
│   ├── entryentropy.py
│   ├── generate_dataset.py
│   ├── humanizeanswers.py
│   └── prune.py
├── src/
│   ├── inference/
│   │   └── infer.py        # Single-sample scoring logic
│   ├── models/
│   │   └── student_scorer.py   # DistilBERT + GRU + Linear architecture
│   ├── training/
│   │   └── train_v2.py     # Training loop (current version)
│   ├── utils/              # Config, logging, text preprocessing
│   └── verification/       # Pipeline integrity checks
├── .gitignore
└── README.md
```

> `checkpoints/`, `data/`, `logs/`, `plots/`, and `venv/` are excluded from version control
> via `.gitignore`. Model weights and datasets must be sourced separately.

---

## Setup

```bash
git clone https://github.com/saipuneeth20/Hybrid_student_assesment.git
cd Hybrid_student_assesment

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

---

## Training

```bash
python src/training/train_v2.py
```

Checkpoints are saved to `checkpoints/` at configurable intervals. Training logs are written
to `logs/`. Configure hyperparameters via `src/utils/config_v2.py`.

---

## Inference

**Single sample via script:**

```bash
python src/inference/infer.py \
  --student_answer "Photosynthesis is the process by which plants make food using sunlight." \
  --reference_answer "Photosynthesis is the process plants use to convert light energy into glucose." \
  --checkpoint checkpoints/student_scorer_v2_777.pt
```

**API demo:**

```bash
cd app
uvicorn main:app --reload
```

Then POST to `http://localhost:8000/score`:

```json
{
  "student_answer": "...",
  "reference_answer": "..."
}
```

Returns:

```json
{
  "score": 78.4
}
```

---

## Dataset

Training data is synthetic, generated to simulate student short-answer responses across
rural school curriculum topics. Real labeled data was unavailable due to access constraints
in the target deployment regions. The synthetic pipeline is in `scripts/generate_dataset.py`,
with humanization and entropy-based pruning handled by `humanizeanswers.py` and `prune.py`.

---

## Results

| Metric        | Value |
|---------------|-------|
| MSE           |       |
| MAE           |       |
| Pearson r     |       |

*(Update after final evaluation run)*

---

## Evaluation Limitations

- Model is trained and evaluated entirely on synthetic data. Generalization to real student
  responses has not been validated.
- Evaluation metrics reflect in-distribution performance; out-of-distribution robustness
  (e.g., regional language interference, misspellings common in rural contexts) is untested.
- Scoring scale (0–100) is a linear rescaling of the model's raw output and does not
  correspond to any rubric-defined grading scheme without further calibration.

---

## Publication

Research paper submitted to **IJETT** (International Journal of Engineering Trends and Technology).

---

## License

MIT
