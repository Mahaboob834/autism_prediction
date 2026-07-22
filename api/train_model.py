"""
Run this ONCE (locally or during deploy build step) to train the model and
save it to disk as model.pkl / scaler.pkl. The Flask app then just loads
these files at startup instead of retraining every time it boots.

Usage: python train_model.py
Requires: autism_data.csv in this same folder.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("autism_data.csv")

    df["age"] = df["age"].fillna(df["age"].median())
    df["gender"] = df["gender"].map({"f": 0, "m": 1})
    df["jundice"] = df["jundice"].map({"no": 0, "yes": 1})
    df["austim"] = df["austim"].map({"no": 0, "yes": 1})
    df["Class/ASD"] = df["Class/ASD"].map({"NO": 0, "YES": 1})

    features = [f"A{i}_Score" for i in range(1, 11)] + ["age", "gender", "jundice", "austim"]
    X = df[features]
    y = df["Class/ASD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(scaler.transform(X_test), y_test)
    print(f"Test accuracy: {accuracy:.2%}")

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Saved model.pkl and scaler.pkl")


if __name__ == "__main__":
    main()