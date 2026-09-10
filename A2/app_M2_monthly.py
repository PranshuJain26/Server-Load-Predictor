# http://127.0.0.1:5003/predict
import joblib
import pandas as pd
import numpy as np
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --- 1. Initialize App and Load Artifacts ---

print("--- Loading M2_monthly XGBoost Model and Data ---")

MODEL_NAME = "A2\\M2_monthly.joblib"
DATA_NAME = "A2\\A2_monthly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("M")
    print(f"Successfully loaded {DATA_NAME}. Last data point: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Error: {e}")
    MODEL = None

# --- 2. Prediction Endpoint ---
def predict():
    """Predicts the next MONTH's call count for API A2."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known_date = DATA.index.max()
        target_timestamp = last_known_date + pd.offsets.MonthEnd(1)

        print(f"Prediction requested for month ending: {target_timestamp.isoformat()}")

        # Positional lag features (correct for monthly)
        lag_3_value = DATA["call_count"].iloc[-3]
        rolling_mean_3 = DATA["call_count"].iloc[-3:].mean()

        features = [
            target_timestamp.month,
            target_timestamp.quarter,
            target_timestamp.year,
            lag_3_value,
            rolling_mean_3
        ]


        feature_vector = np.array(features).reshape(1, -1)
        prediction_raw = MODEL.predict(feature_vector)

        final_prediction = int(np.round(prediction_raw[0]).clip(0))

        return jsonify({
            "api_code": "A2",
            "model_type": "XGBoost_Monthly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

