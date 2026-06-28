from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import pandas as pd

from eeg_processing import extract_features


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =====================
# LOAD MODEL
# =====================

model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# =====================
# HOME
# =====================

@app.route("/")
def home():
    return render_template("index.html")


# =====================
# PREDICT
# =====================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    username = request.form.get("username", "User")


    path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(path)


    # membaca EEG raw TXT
    raw_data = pd.read_csv(
        path,
        header=None
    ).values


    # preprocessing + feature extraction
    features, band_power, clean_signal = extract_features(
        raw_data
    )


    df = pd.DataFrame(
        [features]
    )


    # samakan urutan fitur training
    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )


    prediction = model.predict_proba(df)[0]


    risk_score = float(prediction[1] * 100)



    # =====================
    # DATA GRAFIK EEG
    # =====================

    eeg_plot = {

        "channel1":
        clean_signal[:,0][:500].tolist(),

        "channel2":
        clean_signal[:,1][:500].tolist()

    }


    response = {

        "success": True,

        "name": username,

        "risk_score":
        round(risk_score,2),


        "brainwave":

        band_power,


        "eeg_signal":

        eeg_plot

    }


    return jsonify(response)



if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )