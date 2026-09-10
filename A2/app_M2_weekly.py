# http://127.0.0.1:5002/predict
import joblib
import pandas as pd
import numpy as np
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --- 1. Initialize App and Load Artifacts ---

print("--- Loading M2_weekly XGBoost Model and Data ---")

MODEL_NAME = "A2\\M2_weekly.joblib"
DATA_NAME = "A2\\A2_weekly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("W")
    print(f"Successfully loaded {DATA_NAME}. Last data point: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Error: {e}")
    MODEL = None

# --- 2. Prediction Endpoint ---
def predict():
    """Predicts the next WEEK's call count for API A2."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known_date = DATA.index.max()
        target_timestamp = last_known_date + pd.Timedelta(weeks=1)

        print(f"Prediction requested for week ending: {target_timestamp.isoformat()}")

        lag_4_timestamp = target_timestamp - pd.Timedelta(weeks=4)
        rolling_start = target_timestamp - pd.Timedelta(weeks=4)
        rolling_end = target_timestamp - pd.Timedelta(weeks=1)

        if lag_4_timestamp not in DATA.index:
            return jsonify({"error": "Not enough historical data"}), 400

        features = [
            target_timestamp.month,
            int(target_timestamp.isocalendar().week),
            target_timestamp.quarter,
            DATA.loc[lag_4_timestamp]["call_count"],
            DATA.loc[rolling_start:rolling_end]["call_count"].mean()
        ]

        feature_vector = np.array(features).reshape(1, -1)
        prediction = MODEL.predict(feature_vector)[0]

        return jsonify({
            "api_code": "A2",
            "model_type": "XGBoost_Weekly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": int(round(max(prediction, 0)))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

