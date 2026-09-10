# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
import numpy as np
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------

print("--- Loading M42 (Monthly) MovingAverage Model ---")

MODEL_NAME = "A42\\M42_monthly.joblib"
DATA_NAME = "A42\\A42_monthly_calls.csv"

try:
    ARTIFACT = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("M")

    print(f"Loaded model and data. Last month: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    ARTIFACT = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT MONTH call count for API A42 using Moving Average."""

    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_month = DATA.index.max()
        target_month = last_month + pd.offsets.MonthEnd(1)

        window = ARTIFACT["window"]
        history = ARTIFACT["history"]

        prediction = int(round(max(0, np.mean(history[-window:]))))

        return jsonify({
            "api_code": "A42",
            "model_type": "MovingAverage_Monthly",
            "forecast_for_month": target_month.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


