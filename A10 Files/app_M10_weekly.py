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


print("--- Loading M10 (Weekly) Moving Average Model ---")

MODEL_NAME = "A10\\M10_weekly.joblib"
DATA_NAME = "A10\\A10_weekly_calls.csv"

try:
    ARTIFACT = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("W")

    print(f"Loaded model and data. Last week: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    ARTIFACT = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT WEEK call count for API A10 using Moving Average."""

    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        history = ARTIFACT["history"]
        window = ARTIFACT["window"]

        prediction = int(
            np.round(np.mean(history[-window:])).clip(0)
        )

        last_known_week = DATA.index.max()
        target_week = last_known_week + pd.Timedelta(weeks=1)

        return jsonify({
            "api_code": "A10",
            "model_type": "MovingAverage_Weekly",
            "forecast_for_week": target_week.date().isoformat(),
            "predicted_call_count": prediction,
            "window": window
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

