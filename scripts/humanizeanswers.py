import pandas as pd, random, re

df = pd.read_csv("data/train777_v2_2_final.csv")

fillers = ["like", "basically", "kind of", "you know", "I mean", "sort of"]
hes = ["I am not sure", "maybe", "I guess", "not fully sure"]
drops = ["", "", "", ""]

def humanize(text):
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.06:
            words[i] = random.choice(fillers) + " " + words[i]
        if random.random() < 0.04:
            words[i] = words[i] + " " + random.choice(hes)
        if random.random() < 0.03:
            words[i] = words[i].replace(".", "")
    return " ".join(words)

df["answer"] = df["answer"].apply(humanize)
df.to_csv("data/train777_v2_3_final.csv", index=False)

print("Dataset humanized to V2.3")