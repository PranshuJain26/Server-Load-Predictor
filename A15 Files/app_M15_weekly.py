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

print("--- Loading M15 (Weekly) SARIMA Model ---")

MODEL_NAME = "A15\\M15_weekly.joblib"
DATA_NAME = "A15\\A15_weekly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("W")

    print(f"Loaded model and data. Last week: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT WEEK call count for API A15 using SARIMA."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known_week = DATA.index.max()
        target_week = last_known_week + pd.Timedelta(weeks=1)

        prediction = int(
            np.round(MODEL.forecast(1)[0]).clip(0)
        )

        return jsonify({
            "api_code": "A15",
            "model_type": "SARIMA_Weekly",
            "forecast_for_week": target_week.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )
