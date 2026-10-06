from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("nba_random_forest_model.pkl")

# Exact feature order used during model training
features = joblib.load("nba_features.pkl")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")


# --------------------------------------------------
# INPUT PAGE
# --------------------------------------------------

@app.route("/input")
def input_page():
    return render_template("input.html")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        values = []

        for feature in features:

            value = request.form.get(feature)

            if value is None or value.strip() == "":
                return "Please enter all 10 performance values."

            values.append(float(value))


        # Create dataframe in exact training order
        input_data = pd.DataFrame(
            [values],
            columns=features
        )


        # Make prediction
        prediction = model.predict(input_data)[0]


        # Probability
        probability = model.predict_proba(input_data)[0]

        confidence = max(probability) * 100


        # Result
        if int(prediction) == 1:

            result = "HIGH SCORER"

        else:

            result = "NOT A HIGH SCORER"


        # Dictionary for displaying values
        display_values = {}

        for feature, value in zip(features, values):
            display_values[feature] = value


        return render_template(
            "result.html",
            result=result,
            confidence=round(confidence, 2),
            input_data=display_values
        )


    except ValueError:

        return "Please enter valid numeric values."


if __name__ == "__main__":
    app.run(debug=True)