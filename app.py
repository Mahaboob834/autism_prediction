"""
Flask backend + frontend server for the Autism Screening quiz.

Deployment model:
  - train_model.py is run once (locally, or as a build step) to produce
    model.pkl and scaler.pkl.
  - app.py just LOADS those files at startup (fast, no retraining on boot)
    and serves both the API and the static frontend (static/index.html).

Local run:
  python train_model.py      # produces model.pkl / scaler.pkl
  python app.py               # dev server on http://localhost:5000

Production run (e.g. on Render/Railway/Fly.io):
  gunicorn app:app
"""

import os

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # loosen/tighten this for production - see notes below

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None


def load_model():
    global model, scaler
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise RuntimeError(
            "model.pkl / scaler.pkl not found. Run `python train_model.py` first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


load_model()  # runs once when the process starts (dev server or gunicorn worker)


@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        answers = data["answers"]
        if len(answers) != 10:
            return jsonify({"error": "Expected exactly 10 answers"}), 400

        age = float(data["age"])
        gender = int(data["gender"])
        jaundice = int(data["jaundice"])
        family_asd = int(data["familyASD"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    user_data = list(answers) + [age, gender, jaundice, family_asd]
    user_array = np.array(user_data, dtype=float).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    prediction = int(model.predict(user_scaled)[0])
    probability = float(model.predict_proba(user_scaled)[0][1])
    risk = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"

    return jsonify({"prediction": prediction, "probability": probability, "risk": risk})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
