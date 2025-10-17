from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'. Run train_model.py first to create it."
    )

pipe = joblib.load(MODEL_PATH)

def parse_form(form):
    def get_int(name): return int(form.get(name))
    def get_float(name): return float(form.get(name))
    def get_str(name): return str(form.get(name))

    row = {
        "Gender": get_str("Gender"),
        "Relationship": get_str("Relationship"),
        "No of Days Last Contact": get_int("No_of_Days_Last_Contact"),
        "Personality Type": get_str("Personality_Type"),
        "Confidence": get_float("Confidence"),
        "Empathy Level": get_str("Empathy_Level"),
        "Emotional Stability": get_int("Emotional_Stability"),
        "Trust Parameter": get_int("Trust_Parameter"),
        "No of Past Partners": get_int("No_of_Past_Partners"),
        "No of Past Conflicts": get_int("No_of_Past_Conflicts"),
        "Duration of Relationship": get_int("Duration_of_Relationship"),
        "Level of Closeness": get_int("Level_of_Closeness"),
        "Average Message Response Time in hours": get_float("Average_Message_Response_Time_hours"),
        "Caring": get_int("Caring"),
        "Loving": get_int("Loving"),
        "Efforts": get_int("Efforts"),
        "Age": get_int("Age"),
        "Behaviour": get_str("Behaviour"),
        "Past Relationship Patterns": get_str("Past_Relationship_Patterns"),
        "History of Infidelity": get_int("History_of_Infidelity"),
        "Religion": get_int("Religion"),
        "Time spent together in hours per week": get_float("Time_spent_together_hours_per_week"),
        "Attachment Style": get_str("Attachment_Style"),
        "Body Count": get_int("Body_Count")
    }
    return pd.DataFrame([row])

def apply_domain_rules(input_row: pd.Series, happy: float, cheat: float):
    # Read fields
    body_count = int(input_row["Body Count"])
    history_infidelity = int(input_row["History of Infidelity"])
    attachment = str(input_row["Attachment Style"]).lower()
    past_pattern = str(input_row["Past Relationship Patterns"]).lower()
    trust = int(input_row["Trust Parameter"])
    behaviour = str(input_row["Behaviour"]).lower()
    time_week = float(input_row["Time spent together in hours per week"])

    # --- Hard rules / user instructions ---
    # Past pattern & attachment exact effects
    if past_pattern == "casual":
        cheat += 4.0
    if attachment == "low":
        cheat += 3.0
    if attachment == "high":
        cheat -= 20.0
        happy += 12.0

    # Body count thresholds
    if body_count > 20:
        cheat = max(cheat, 90.0)
        happy -= 30.0
    elif body_count > 10:
        cheat = max(cheat, 50.0)
    elif body_count < 3:
        cheat -= 10.0

    # Prior infidelity still a strong factor
    if history_infidelity > 0:
        cheat += 20.0
        happy -= 15.0

    # Attachment generally: more attachment => less cheating, more happy
    # (already handled for high/low explicitly)
    # Gentle positives
    if trust >= 9:
        happy += 3.0
        cheat -= 2.0
    if behaviour == "kind":
        cheat -= 2.0
        happy += 2.0
    if time_week >= 25:
        happy += 2.0
        cheat -= 1.0

    # Clamp
    happy = float(np.clip(happy, 1.0, 99.0))
    cheat = float(np.clip(cheat, 1.0, 99.0))
    return happy, cheat

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    X = parse_form(request.form)
    y = pipe.predict(X)[0]
    happy = float(np.clip(y[0], 1, 99))
    cheat = float(np.clip(y[1], 1, 99))

    # Apply domain rules
    happy, cheat = apply_domain_rules(X.iloc[0], happy, cheat)
    return render_template("result.html", happy=round(happy,1), cheat=round(cheat,1))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
