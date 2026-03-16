# app/classify.py

def classify_score(score: float):
    if score < 40:
        return "Weak"
    elif score < 75:
        return "Moderate"
    else:
        return "Strong"