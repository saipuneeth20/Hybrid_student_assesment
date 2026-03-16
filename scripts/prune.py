import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("data/train777.csv")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

MAX = 500

def prune(text):
    sentences = [s.strip() for s in str(text).split(".") if len(s.strip())>3]
    seen = set()
    cleaned = []

    for s in sentences:
        key = s.lower()
        if key not in seen:
            cleaned.append(s)
            seen.add(key)

    pruned = ". ".join(cleaned) + "."

    tokens = tok.encode(pruned, truncation=False)
    if len(tokens) <= MAX:
        return pruned

    # If still long, trim only trailing filler
    while len(tokens) > MAX and len(cleaned) > 3:
        cleaned.pop()   # remove last filler sentence
        pruned = ". ".join(cleaned) + "."
        tokens = tok.encode(pruned, truncation=False)

    return pruned

df["answer"] = df["answer"].apply(prune)
df.to_csv("data/train777_bert_safe_semantic.csv", index=False)

print("Semantic-safe dataset created.")
