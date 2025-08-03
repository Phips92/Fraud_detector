from flask import Flask, request, jsonify, abort
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Security mechanism with API-Key
API_KEY = "geht_di_nix_an"

@app.before_request
def restrict_access():
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        abort(403)

# Load model and preprocessing artifacts
model = joblib.load("fraud_model.joblib")
scaler = joblib.load("scaler.joblib")
merchant_cols = joblib.load("merchant_columns.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Required fields check
    required_fields = [
        "amt", "city_pop", "merch_lat", "merch_long", 
        "merchant", "dob", "trans_date_trans_time"
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Create input DataFrame
    df_input = pd.DataFrame([data])

    # Type validation
    try:
        df_input["amt"] = df_input["amt"].astype(float)
        df_input["city_pop"] = df_input["city_pop"].astype(int)
        df_input["merch_lat"] = df_input["merch_lat"].astype(float)
        df_input["merch_long"] = df_input["merch_long"].astype(float)
    except Exception as e:
        return jsonify({"error": f"Invalid input types: {e}"}), 400

    # Age computation
    try:
        df_input["dob"] = pd.to_datetime(df_input["dob"])
        df_input["trans_date_trans_time"] = pd.to_datetime(df_input["trans_date_trans_time"])
        df_input["age"] = (df_input["trans_date_trans_time"] - df_input["dob"]).dt.days // 365
    except Exception as e:
        return jsonify({"error": f"Invalid datetime format: {e}"}), 400

    # Feature scaling
    try:
        df_input["transaction_amount_scaled"] = scaler.transform(df_input[["amt"]])
    except Exception as e:
        return jsonify({"error": f"Scaler failed: {e}"}), 400

    # One-hot encoding for merchant
    df_input = pd.get_dummies(df_input, columns=["merchant"], prefix="merchant")
    for col in merchant_cols:
        if col not in df_input:
            df_input[col] = 0

    # Ensure correct column order
    feature_cols = [
        "transaction_amount_scaled", "city_pop", "merch_lat", "merch_long", "age"
    ] + merchant_cols
    df_input = df_input[feature_cols]

    # Predict probability
    try:
        proba = model.predict_proba(df_input)[:, 1][0]
        return jsonify({"fraud_probability": round(float(proba), 4)})
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

