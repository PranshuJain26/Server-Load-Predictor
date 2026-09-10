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


print("--- Loading M69 (Monthly) Moving Average Model ---")

MODEL_NAME = "A69\\M69_monthly.joblib"
DATA_NAME = "A69\\A69_monthly_calls.csv"

try:
    MODEL_OBJ = joblib.load(MODEL_NAME)
    WINDOW = MODEL_OBJ["window"]
    HISTORY = MODEL_OBJ["history"]

    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("M")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL_OBJ = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT MONTH call count for API A69 using Moving Average."""

    if MODEL_OBJ is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.offsets.MonthEnd(1)

        prediction = int(
            max(0, round(np.mean(HISTORY[-WINDOW:])))
        )

        return jsonify({
            "api_code": "A69",
            "model_type": "MovingAverage_Monthly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

