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


print("--- Loading M45 (Monthly) MovingAverage Model ---")

MODEL_NAME = "A45\\M45_monthly.joblib"
DATA_NAME = "A45\\A45_monthly_calls.csv"

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
    """Predict NEXT MONTH call count for API A45 using Moving Average."""

    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        history = ARTIFACT["history"]
        window = ARTIFACT["window"]

        prediction = int(
            max(0, round(np.mean(history[-window:])))
        )

        last_month = DATA.index.max()
        target_month = last_month + pd.offsets.MonthEnd(1)

        return jsonify({
            "api_code": "A45",
            "model_type": "MovingAverage_Monthly",
            "forecast_for_month": target_month.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


