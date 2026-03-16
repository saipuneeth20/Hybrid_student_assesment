import pandas as pd, random

df = pd.read_csv("data/train777_v2_1_final.csv")

openings = [
    "When I first read this question, I thought about how this concept is used in real life.",
    "This question confused me at first, but then I tried to remember what we studied.",
    "An example that comes to my mind immediately is related to this idea.",
    "The main point here is something we discussed many times in class.",
    "I want to start by saying what I remember from the lesson.",
    "In everyday situations, this concept can actually be noticed.",
    "The first thing my teacher explained about this was the basic idea.",
    "I usually understand this topic by thinking of an example.",
    "At first glance this seems simple, but it actually needs explanation.",
    "This is one of those questions where the final result comes first.",
    "A mistake students often make in this topic is misunderstanding the base idea.",
    "The situation described in the question connects to what we studied earlier.",
    "To understand this, we have to look at what happens step by step.",
    "There is a small detail in this topic that changes the full meaning.",
    "I remember struggling with this topic when it was first taught.",
    "This topic can be linked to another chapter we studied earlier.",
    "The idea behind this question becomes clear after thinking about an example.",
    "I will try to explain this based on what I could recall.",
    "At the beginning, I was not sure what exactly the question wanted.",
    "Before answering, I tried to imagine how this works in practice.",
    "The reason this works can be understood by thinking about its effect.",
    "One common confusion about this topic is mixing it with another idea.",
    "If we look at the effect first, the cause becomes clearer.",
    "It helps to think about this by breaking it into small parts.",
    "A real world situation makes this easier to understand."
]

def replace_opening(ans):
    parts = ans.split(".", 1)
    if len(parts) == 2:
        return random.choice(openings) + ". " + parts[1].strip()
    else:
        return random.choice(openings) + ". " + ans

df["answer"] = df["answer"].apply(replace_opening)
df.to_csv("data/train777_v2_2_final.csv", index=False)

print("Dataset upgraded to V2.2")
