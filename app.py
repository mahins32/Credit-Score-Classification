# Importing Libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loading models
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lr_model = pickle.load(open("log_reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["f1"]),
            float(request.form["f2"]),
            float(request.form["f3"]),
            float(request.form["f4"])
        ]

        model_choice = request.form["model"]

        # Scale features
        final_features = scaler.transform([features])

        # Prediction
        if model_choice == "rf":
            prediction = rf_model.predict(final_features)[0]
            model_used = "Random Forest"
        else:
            prediction = lr_model.predict(final_features)[0]
            model_used = "Logistic Regression"

        return render_template(
            "index.html",
            prediction=prediction,
            model_used=model_used
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
