# scripts/rebuild_dataset.py
# Run: python scripts/rebuild_dataset.py

import re
import pandas as pd

INPUT_PATH = "data/train_combined.csv"
OUTPUT_PATH = "data/train_clean.csv"


def normalize_question(q: str) -> str:
    if not isinstance(q, str):
        return ""
    q = q.strip().lower()
    q = re.sub(r'\s+', ' ', q)
    q = q.replace(" ?", "?").rstrip("?").strip()
    return q


def main():

    # -----------------------------------------------
    # Load
    # -----------------------------------------------
    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.str.strip().str.lower()

    print("Columns found:", list(df.columns))
    print("Total rows:", len(df))

    # -----------------------------------------------
    # Clean columns
    # -----------------------------------------------
    for col in ["answer", "answers", "context", "question"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    reference_rows = df[df["answer"] != ""].copy()
    student_rows   = df[df["answers"] != ""].copy()

    print(f"Reference rows (answer column): {len(reference_rows)}")
    print(f"Student answer rows (answers column): {len(student_rows)}")

    # -----------------------------------------------
    # Normalize questions
    # -----------------------------------------------
    reference_rows["norm_q"] = reference_rows["question"].apply(normalize_question)
    student_rows["norm_q"]   = student_rows["question"].apply(normalize_question)

    # -----------------------------------------------
    # Build reference lookup
    # -----------------------------------------------
    ref_lookup = (
        reference_rows
        .groupby("norm_q")
        .first()[["answer"]]
        .rename(columns={"answer": "reference_answer"})
    )

    print(f"Unique reference questions: {len(ref_lookup)}")

    # Manual references for ICT questions missing from student 1 data
    manual_refs = {
        "what are the basic uses of a computer":        "computers are used for calculations communication education and entertainment",
        "what are the basic characteristics of a computer": "computers are fast accurate reliable and can store large amounts of data",
        "what are input devices":                       "input devices are hardware used to send data to a computer such as keyboard and mouse",
        "what are output devices":                      "output devices display or produce results from a computer such as monitor and printer",
        "what are the functions of an operating system": "an operating system manages hardware resources and provides services for programs",
        "what are the types of operating systems":      "types of operating systems include batch time sharing distributed and real time",
        "what is the internet":                         "the internet is a global network of computers connected to share information",
        "what are the uses of the internet":            "the internet is used for communication education shopping and entertainment",
        "what are the different types of the computer networks": "types of computer networks include lan wan and man",
        "what are the advantages of computer networks": "computer networks allow resource sharing communication and data transfer",
        "what are the cyber safety passwords":          "cyber safety passwords should be strong unique and kept confidential",
        "what cyber safety":                            "cyber safety means protecting personal information and staying safe online",
        "change one sentence into passive voice":       "in passive voice the subject receives the action of the verb",
    }

    manual_ref_df = pd.DataFrame([
        {"norm_q": k, "reference_answer": v}
        for k, v in manual_refs.items()
    ]).set_index("norm_q")

    ref_lookup = pd.concat([
        ref_lookup,
        manual_ref_df[~manual_ref_df.index.isin(ref_lookup.index)]
    ])

    # -----------------------------------------------
    # Join student rows with reference answers
    # -----------------------------------------------
    student_rows = student_rows.join(ref_lookup, on="norm_q", how="left")
    student_rows["reference_answer"] = student_rows["reference_answer"].fillna("")
    student_rows["context"] = student_rows["context"].fillna("").astype(str).str.strip()

    no_ref = student_rows[student_rows["reference_answer"] == ""]
    if len(no_ref) > 0:
        print(f"WARNING: {len(no_ref)} student rows had no matching reference — dropping them")
        print("Unmatched questions:", no_ref["question"].unique())

    student_rows = student_rows[student_rows["reference_answer"] != ""].copy()

    # -----------------------------------------------
    # Build clean dataframe
    # -----------------------------------------------
    clean_df = pd.DataFrame({
        "question":         student_rows["question"].values,
        "context":          student_rows["context"].values,
        "reference_answer": student_rows["reference_answer"].values,
        "student_answer":   student_rows["answers"].values,
        "score":            pd.to_numeric(student_rows["score"], errors="coerce")
    })

    before = len(clean_df)
    clean_df = clean_df.dropna(subset=["score"]).reset_index(drop=True)
    clean_df = clean_df[clean_df["student_answer"].str.strip() != ""].reset_index(drop=True)
    print(f"Dropped {before - len(clean_df)} rows with missing scores or empty student answers")

    clean_df["score"] = clean_df["score"].clip(0, 100)

    print(f"\nClean dataset before augmentation: {len(clean_df)} rows")
    print("\nScore distribution (before augmentation):")
    print(clean_df["score"].describe())

    # -----------------------------------------------
    # Augmentation — contrastive pairs across full score range
    # -----------------------------------------------
    augmented = [

        # ── ENGLISH: What is a noun? ──
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "a noun is a word that names a person place or thing", "score": 88.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "noun identifies a person or place", "score": 60.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "noun is a naming word", "score": 45.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "it is a word", "score": 15.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "i dont know what noun is", "score": 5.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "i like playing cricket", "score": 3.0},

        # ── ENGLISH: What is a verb? ──
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "a verb is a word that shows action or state of being in a sentence", "score": 92.0},
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "a verb shows action", "score": 65.0},
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "verb is doing word", "score": 40.0},
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "verb is something in english grammar", "score": 12.0},
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "i think it is a noun", "score": 4.0},
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "the weather is nice today", "score": 2.0},

        # ── ENGLISH: Present tense ──
        {"question": "What is present tense?", "context": "", "reference_answer": "present tense is used to describe actions happening now or regularly", "student_answer": "present tense describes actions that are happening now or occur regularly", "score": 90.0},
        {"question": "What is present tense?", "context": "", "reference_answer": "present tense is used to describe actions happening now or regularly", "student_answer": "present tense is about current actions", "score": 62.0},
        {"question": "What is present tense?", "context": "", "reference_answer": "present tense is used to describe actions happening now or regularly", "student_answer": "it talks about now", "score": 35.0},
        {"question": "What is present tense?", "context": "", "reference_answer": "present tense is used to describe actions happening now or regularly", "student_answer": "tense is a grammar thing", "score": 10.0},
        {"question": "What is present tense?", "context": "", "reference_answer": "present tense is used to describe actions happening now or regularly", "student_answer": "i went to school yesterday", "score": 3.0},

        # ── ENGLISH: Past vs present tense ──
        {"question": "Explain the difference between past and present tense", "context": "", "reference_answer": "past tense describes completed actions while present tense describes current or ongoing actions", "student_answer": "past tense is for finished actions and present tense is for actions happening now", "score": 88.0},
        {"question": "Explain the difference between past and present tense", "context": "", "reference_answer": "past tense describes completed actions while present tense describes current or ongoing actions", "student_answer": "past is old present is now", "score": 38.0},
        {"question": "Explain the difference between past and present tense", "context": "", "reference_answer": "past tense describes completed actions while present tense describes current or ongoing actions", "student_answer": "they are both tenses in english", "score": 12.0},
        {"question": "Explain the difference between past and present tense", "context": "", "reference_answer": "past tense describes completed actions while present tense describes current or ongoing actions", "student_answer": "i do not know the difference", "score": 4.0},

        # ── SCIENCE: Photosynthesis ──
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "photosynthesis is the process where plants use sunlight to make their own food", "score": 91.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "plants make food using sunlight", "score": 68.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "plants need water and soil", "score": 22.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "it is a biology topic", "score": 10.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "plants drink water from the ground", "score": 5.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "i like science class", "score": 2.0},

        # ── SCIENCE: States of matter ──
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "the three states of matter are solid liquid and gas", "score": 95.0},
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "solid liquid and gas", "score": 80.0},
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "solid and liquid", "score": 45.0},
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "matter is everywhere around us", "score": 15.0},
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "water is wet", "score": 5.0},

        # ── SCIENCE: Speed ──
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "speed is the distance covered per unit of time", "score": 92.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "speed is distance divided by time", "score": 85.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "speed is how fast something moves", "score": 42.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "cars go fast on roads", "score": 8.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "i dont know", "score": 3.0},

        # ── SCIENCE: Acceleration ──
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "acceleration is the rate of change of velocity of an object", "score": 93.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "acceleration means change in velocity", "score": 70.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "acceleration is related to speed", "score": 35.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "acceleration is speed", "score": 20.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "it is a physics term", "score": 8.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "i like football", "score": 2.0},

        # ── SCIENCE: Newton's first law ──
        {"question": "State Newton's first law of motion", "context": "", "reference_answer": "a body remains in rest or motion unless acted upon by an external force", "student_answer": "an object stays at rest or in motion unless an external force acts on it", "score": 90.0},
        {"question": "State Newton's first law of motion", "context": "", "reference_answer": "a body remains in rest or motion unless acted upon by an external force", "student_answer": "objects stay still unless a force moves them", "score": 55.0},
        {"question": "State Newton's first law of motion", "context": "", "reference_answer": "a body remains in rest or motion unless acted upon by an external force", "student_answer": "newton said something about force and motion", "score": 20.0},
        {"question": "State Newton's first law of motion", "context": "", "reference_answer": "a body remains in rest or motion unless acted upon by an external force", "student_answer": "newton was a scientist", "score": 8.0},
        {"question": "State Newton's first law of motion", "context": "", "reference_answer": "a body remains in rest or motion unless acted upon by an external force", "student_answer": "i like science", "score": 2.0},

        # ── SCIENCE: Respiration ──
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "respiration is the process by which organisms release energy from food", "score": 92.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "respiration releases energy from food", "score": 72.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "breathing is respiration", "score": 35.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "we breathe air to live", "score": 18.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "it is a life process", "score": 10.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "i had lunch today", "score": 2.0},

        # ── SCIENCE: Nutrition ──
        {"question": "What is nutrition?", "context": "", "reference_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "student_answer": "nutrition is the process of taking in food and using it for growth and energy", "score": 91.0},
        {"question": "What is nutrition?", "context": "", "reference_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "student_answer": "nutrition is about getting food for energy", "score": 65.0},
        {"question": "What is nutrition?", "context": "", "reference_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "student_answer": "nutrition means eating healthy food", "score": 38.0},
        {"question": "What is nutrition?", "context": "", "reference_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "student_answer": "eating food daily", "score": 18.0},
        {"question": "What is nutrition?", "context": "", "reference_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "student_answer": "i eat rice and dal", "score": 5.0},

        # ── MATHS: Rational number ──
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "a rational number is one that can be expressed as p divided by q where q is not equal to zero", "score": 94.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "a number that can be written as a fraction", "score": 60.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "numbers like half and quarter", "score": 35.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "a number that is rational", "score": 15.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "maths is difficult", "score": 3.0},

        # ── MATHS: Mean ──
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "mean is the sum of all values divided by the number of values", "score": 93.0},
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "mean is the average of given values", "score": 78.0},
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "mean is the middle value", "score": 30.0},
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "mean is a maths word", "score": 10.0},
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "i like numbers", "score": 2.0},

        # ── MATHS: Linear equation ──
        {"question": "What is a linear equation?", "context": "", "reference_answer": "a linear equation is an equation with one variable of degree one", "student_answer": "a linear equation is an equation with one variable raised to the power of one", "score": 91.0},
        {"question": "What is a linear equation?", "context": "", "reference_answer": "a linear equation is an equation with one variable of degree one", "student_answer": "linear equation has one variable", "score": 58.0},
        {"question": "What is a linear equation?", "context": "", "reference_answer": "a linear equation is an equation with one variable of degree one", "student_answer": "x plus y equals something", "score": 25.0},
        {"question": "What is a linear equation?", "context": "", "reference_answer": "a linear equation is an equation with one variable of degree one", "student_answer": "an equation in maths", "score": 18.0},
        {"question": "What is a linear equation?", "context": "", "reference_answer": "a linear equation is an equation with one variable of degree one", "student_answer": "equations are hard", "score": 5.0},

        # ── MATHS: Polynomial ──
        {"question": "What is a polynomial?", "context": "", "reference_answer": "a polynomial is an expression with variables and constants combined using mathematical operations", "student_answer": "a polynomial is a mathematical expression made up of variables constants and operations", "score": 90.0},
        {"question": "What is a polynomial?", "context": "", "reference_answer": "a polynomial is an expression with variables and constants combined using mathematical operations", "student_answer": "polynomial has variables and numbers", "score": 58.0},
        {"question": "What is a polynomial?", "context": "", "reference_answer": "a polynomial is an expression with variables and constants combined using mathematical operations", "student_answer": "it is an algebra expression", "score": 28.0},
        {"question": "What is a polynomial?", "context": "", "reference_answer": "a polynomial is an expression with variables and constants combined using mathematical operations", "student_answer": "polynomial is a maths term", "score": 10.0},
        {"question": "What is a polynomial?", "context": "", "reference_answer": "a polynomial is an expression with variables and constants combined using mathematical operations", "student_answer": "i dont understand maths", "score": 3.0},

        # ── SOCIAL: Democracy ──
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "democracy is a system of government where people choose their leaders through elections", "score": 92.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "democracy is government by the people", "score": 72.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "democracy means freedom", "score": 35.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "it is a type of government", "score": 20.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "india is a country", "score": 5.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "i dont know", "score": 2.0},

        # ── SOCIAL: French Revolution ──
        {"question": "State one cause of the French Revolution", "context": "", "reference_answer": "unfair taxation of the common people while the clergy and nobility were exempt was a major cause", "student_answer": "the common people were burdened with heavy taxes while the rich were exempt", "score": 88.0},
        {"question": "State one cause of the French Revolution", "context": "", "reference_answer": "unfair taxation of the common people while the clergy and nobility were exempt was a major cause", "student_answer": "unfair taxes caused the revolution", "score": 70.0},
        {"question": "State one cause of the French Revolution", "context": "", "reference_answer": "unfair taxation of the common people while the clergy and nobility were exempt was a major cause", "student_answer": "people were angry at the king", "score": 38.0},
        {"question": "State one cause of the French Revolution", "context": "", "reference_answer": "unfair taxation of the common people while the clergy and nobility were exempt was a major cause", "student_answer": "france had a revolution long ago", "score": 12.0},
        {"question": "State one cause of the French Revolution", "context": "", "reference_answer": "unfair taxation of the common people while the clergy and nobility were exempt was a major cause", "student_answer": "i dont know about france", "score": 3.0},

        # ── ICT: Computer ──
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "a computer is an electronic device that processes data according to instructions", "score": 90.0},
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "a computer is a machine that processes information", "score": 72.0},
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "a machine used for work", "score": 35.0},
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "computer is used to play games and watch videos", "score": 22.0},
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "a machine", "score": 12.0},
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "i use computer at school", "score": 5.0},

        # ── ICT: Internet ──
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "the internet is a worldwide network that connects computers to share information", "score": 90.0},
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "internet connects computers around the world", "score": 68.0},
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "it connects phones and laptops", "score": 25.0},
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "internet is used for watching youtube", "score": 20.0},
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "internet is good for students", "score": 12.0},
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "i dont use internet", "score": 3.0},

        # ── SCIENCE: Melting ──
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "melting is when a solid changes into liquid on heating", "score": 93.0},
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "solid becomes liquid when heated", "score": 75.0},
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "ice melts into water", "score": 50.0},
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "melting is related to heat", "score": 22.0},
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "it is a change", "score": 8.0},
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "i dont know", "score": 2.0},

        # ── SOCIAL: Constitution ──
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "a constitution is the supreme law that lays down rules for governing a country", "score": 91.0},
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "constitution is a set of rules for a country", "score": 72.0},
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "constitution has laws", "score": 40.0},
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "it is an important document", "score": 18.0},
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "constitution is like a book", "score": 8.0},
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "i dont know", "score": 2.0},

        # ── MATHS: Straight angle ──
        {"question": "What is a straight angle?", "context": "", "reference_answer": "a straight angle measures exactly 180 degrees and forms a straight line", "student_answer": "a straight angle is an angle that measures 180 degrees forming a straight line", "score": 95.0},
        {"question": "What is a straight angle?", "context": "", "reference_answer": "a straight angle measures exactly 180 degrees and forms a straight line", "student_answer": "straight angle is 180 degrees", "score": 80.0},
        {"question": "What is a straight angle?", "context": "", "reference_answer": "a straight angle measures exactly 180 degrees and forms a straight line", "student_answer": "it is a type of angle", "score": 20.0},
        {"question": "What is a straight angle?", "context": "", "reference_answer": "a straight angle measures exactly 180 degrees and forms a straight line", "student_answer": "angle is in geometry", "score": 10.0},
        {"question": "What is a straight angle?", "context": "", "reference_answer": "a straight angle measures exactly 180 degrees and forms a straight line", "student_answer": "i dont study geometry", "score": 2.0},
        
        # ── TOPIC-RELEVANT LOW SCORES vs COMPLETELY IRRELEVANT ──
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "it is a word in english", "score": 18.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "i like watching movies in the evening", "score": 2.0},
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "yesterday i went to the market with my mother and bought vegetables", "score": 2.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "it is related to plants", "score": 20.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "i like playing cricket with friends", "score": 2.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "it is a type of rule", "score": 22.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "i went to school today and had lunch", "score": 2.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "it is related to motion", "score": 20.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "my favourite food is biryani", "score": 2.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "it is a physics concept", "score": 18.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "i watched a movie last night", "score": 2.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "it is a type of number", "score": 15.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "i like summer holidays", "score": 2.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "it is a body process", "score": 18.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "today the weather is very hot outside", "score": 2.0},

        # ── LONG WRONG ANSWERS — explicitly low ──
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "i really enjoy going to school every day and playing with my friends during lunch break in the evening", "score": 2.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "i woke up early today had breakfast watched television and then went outside to play cricket with my friends for hours", "score": 2.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "my favourite subject is science because i like doing experiments and learning about animals plants and the solar system every week", "score": 2.0},

        # ── SHORT CORRECT ANSWERS — explicitly high ──
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "solid liquid gas", "score": 78.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "distance per unit time", "score": 72.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "government by the people", "score": 68.0},
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "average of values", "score": 60.0},
        # ── PERFECT MATCH PAIRS — score should be 95-100 ──
        {"question": "What is a noun?", "context": "", "reference_answer": "a noun is a naming word used to identify a person place or thing", "student_answer": "a noun is a naming word used to identify a person place or thing", "score": 98.0},
        {"question": "What is a verb?", "context": "", "reference_answer": "a verb is a word that shows action or state of being", "student_answer": "a verb is a word that shows action or state of being", "score": 98.0},
        {"question": "What is speed?", "context": "", "reference_answer": "speed is distance travelled per unit time", "student_answer": "speed is distance travelled per unit time", "score": 98.0},
        {"question": "What is acceleration?", "context": "", "reference_answer": "acceleration is rate of change of velocity", "student_answer": "acceleration is rate of change of velocity", "score": 98.0},
        {"question": "What is photosynthesis?", "context": "photosynthesis is the process by which plants make food using sunlight", "reference_answer": "photosynthesis is process by which plants make food using sunlight", "student_answer": "photosynthesis is the process by which plants make food using sunlight", "score": 98.0},
        {"question": "What is democracy?", "context": "", "reference_answer": "democracy is a form of government by the people where citizens elect their representatives", "student_answer": "democracy is a form of government by the people where citizens elect their representatives", "score": 98.0},
        {"question": "What is a rational number?", "context": "", "reference_answer": "a rational number can be written as p by q where q is not zero", "student_answer": "a rational number can be written as p by q where q is not zero", "score": 98.0},
        {"question": "What is respiration?", "context": "", "reference_answer": "respiration is process of releasing energy from food in living organisms", "student_answer": "respiration is process of releasing energy from food in living organisms", "score": 98.0},
        {"question": "What are the three states of matter?", "context": "", "reference_answer": "solid liquid and gas are the three states of matter", "student_answer": "solid liquid and gas are the three states of matter", "score": 98.0},
        {"question": "What is the mean of data?", "context": "", "reference_answer": "mean is the average of given values calculated by dividing sum by count", "student_answer": "mean is the average of given values calculated by dividing sum by count", "score": 98.0},
        {"question": "What is a computer?", "context": "", "reference_answer": "a computer is an electronic machine that processes data and performs calculations", "student_answer": "a computer is an electronic machine that processes data and performs calculations", "score": 98.0},
        {"question": "What is the internet?", "context": "", "reference_answer": "the internet is a global network of computers connected to share information and communicate", "student_answer": "the internet is a global network of computers connected to share information and communicate", "score": 98.0},
        {"question": "What is melting?", "context": "", "reference_answer": "melting is the change of a solid into a liquid when heated", "student_answer": "melting is the change of a solid into a liquid when heated", "score": 98.0},
        {"question": "What is nutrition?", "context": "", "reference_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "student_answer": "nutrition is the process of obtaining and using food for energy growth and body functions", "score": 98.0},
        {"question": "What is a constitution?", "context": "", "reference_answer": "a constitution is a set of fundamental rules and principles that govern a country", "student_answer": "a constitution is a set of fundamental rules and principles that govern a country", "score": 98.0},
        ]

    aug_df = pd.DataFrame(augmented)
    clean_df = pd.concat([clean_df, aug_df], ignore_index=True)
    clean_df["score"] = clean_df["score"].clip(0, 100)

    print(f"\nAfter augmentation: {len(clean_df)} rows")
    print("\nFinal score distribution:")
    print(clean_df["score"].describe())

    # -----------------------------------------------
    # Save
    # -----------------------------------------------
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()