import pandas as pd
import random
from pathlib import Path

df = pd.read_csv("data/train777_v1_1.csv")

def expand_answer(a, subject, style):
    a = str(a).strip()

    s1 = f"I think the answer is {a.lower()}."
    s2 = f"In {subject}, we studied this topic and I remember learning about it in class."
    s3 = f"This concept works in a certain way because it is connected to other ideas in {subject}."
    s4 = f"When this happens, certain steps occur which show how the process actually takes place."
    s5 = f"These steps explain why {a.lower()} is considered correct in most situations."
    s6 = f"Sometimes I might be missing some details or mixing it with another topic."
    s7 = f"So this is the explanation I could write based on my understanding."

    if style == 0:
        return " ".join([s1, s2, s3, s4, s5, s7])
    if style == 1:
        return " ".join([s1, s2, s3, s4, s5, s6, s7])
    if style == 2:
        return " ".join([s1, s3, s4, s7])
    if style == 3:
        return " ".join([s1, s2, s4, s6, s7])
    if style == 4:
        return " ".join([s1, s2, s3, s6, s7])
    if style == 5:
        return " ".join([s1, s5, s4, s3, s2, s7])
    if style == 6:
        return " ".join([s1, s2, s3, s4, s5, s6, s7])
    if style == 7:
        return " ".join([s1, s2, s4, s5, s7])
    if style == 8:
        return " ".join([s1, s3, s5, s6, s7])
    if style == 9:
        return " ".join([s1, s2, s3, s4, s7])
    if style == 10:
        return " ".join([s1, s2, s5, s7])
    if style == 11:
        return " ".join([s1, s3, s4, s5, s6, s7])
    if style == 12:
        return " ".join([s1, s2, s3, s6, s7])
    if style == 13:
        return " ".join([s1, s2, s4, s6, s7])
    if style == 14:
        return " ".join([s1, s3, s4, s6, s7])
    if style == 15:
        return " ".join([s5, s4, s3, s2, s1, s7])
    if style == 16:
        return " ".join([s1, s2, s3, s4])
    if style == 17:
        return " ".join([s1, s1, s2, s3, s4, s7])

def generate(row):
    return expand_answer(row["answer"], row["subject"], random.randint(0,17))

df["answer"] = df.apply(generate, axis=1)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/train777_v2_hybrid_studentPOV.csv", index=False)
print("Hybrid student-POV dataset created:", len(df))
