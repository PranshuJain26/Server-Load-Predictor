# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify

import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------


print("--- Loading M69 (Weekly) SARIMA Model ---")

MODEL_NAME = "A69\\M69_weekly.joblib"
DATA_NAME = "A69\\A69_weekly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("W")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------

def predict():
    """Predict NEXT WEEK call count for API A69 using SARIMA."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.Timedelta(weeks=1)

        prediction = int(
            max(0, round(MODEL.forecast(1)[0]))
        )

        return jsonify({
            "api_code": "A69",
            "model_type": "SARIMA_Weekly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


