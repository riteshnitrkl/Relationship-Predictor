import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

def apply_label_rules(df: pd.DataFrame):
    df = df.copy()

    # --- Core instructions provided by user ---
    # Attachment rules
    df.loc[df["Attachment Style"].str.lower() == "low", "Chances of Cheating %"] += 3
    df.loc[df["Attachment Style"].str.lower() == "high", "Chances of Cheating %"] -= 15
    df.loc[df["Attachment Style"].str.lower() == "high", "Chances of Happy Marriage %"] += 8

    # Past pattern
    df.loc[df["Past Relationship Patterns"].str.lower() == "casual", "Chances of Cheating %"] += 4

    # Body count hard constraints / priors
    df.loc[df["Body Count"] > 20, "Chances of Cheating %"] = (
        df.loc[df["Body Count"] > 20, "Chances of Cheating %"].clip(lower=90)
    )
    df.loc[df["Body Count"] > 20, "Chances of Happy Marriage %"] -= 30
    df.loc[(df["Body Count"] > 10) & (df["Body Count"] <= 20), "Chances of Cheating %"] = (
        df.loc[(df["Body Count"] > 10) & (df["Body Count"] <= 20), "Chances of Cheating %"].clip(lower=50)
    )
    df.loc[df["Body Count"] < 3, "Chances of Cheating %"] -= 10  # low cheating for low body count

    # Prior infidelity
    df.loc[df["History of Infidelity"] > 0, "Chances of Cheating %"] += 20
    df.loc[df["History of Infidelity"] > 0, "Chances of Happy Marriage %"] -= 15

    # Trust & stability gentle effects
    df.loc[df["Trust Parameter"] >= 9, "Chances of Happy Marriage %"] += 4
    df.loc[df["Trust Parameter"] <= 3, "Chances of Cheating %"] += 6

    df.loc[df["Emotional Stability"] >= 8, "Chances of Happy Marriage %"] += 3
    df.loc[df["Emotional Stability"] <= 3, "Chances of Cheating %"] += 8
    df.loc[df["Emotional Stability"] <= 3, "Chances of Happy Marriage %"] -= 6

    # Caring/Loving/Efforts small boosts to happy
    df.loc[:, "Chances of Happy Marriage %"] = (
        df["Chances of Happy Marriage %"] + 0.1 * (df["Caring"] - 5)
    )
    df.loc[:, "Chances of Happy Marriage %"] = (
        df["Chances of Happy Marriage %"] + 0.1 * (df["Loving"] - 5)
    )
    df.loc[:, "Chances of Happy Marriage %"] = (
        df["Chances of Happy Marriage %"] + 0.15 * (df["Efforts"] - 5)
    )

    # Clip ranges
    df.loc[:, "Chances of Cheating %"] = df["Chances of Cheating %"].clip(1, 99)
    df.loc[:, "Chances of Happy Marriage %"] = df["Chances of Happy Marriage %"].clip(1, 99)

    return df

def train_model(data_path):
    df = pd.read_csv(data_path)
    df = apply_label_rules(df)

    X = df.drop(columns=["Chances of Happy Marriage %", "Chances of Cheating %"])
    y = df[["Chances of Happy Marriage %", "Chances of Cheating %"]]

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=450,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1
    ))
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, "pipeline.pkl")
    print("✅ Trained with domain rules. Saved pipeline.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train_model(args.data)