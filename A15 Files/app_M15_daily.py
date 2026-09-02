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

print("--- Loading M15 (Daily) SARIMA Model ---")

MODEL_NAME = "A15\\M15_daily.joblib"
DATA_NAME = "A15\\A15_daily_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("D")

    print(f"Loaded model and data. Last date: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT DAY call count for API A15 using SARIMA."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known_date = DATA.index.max()
        target_date = last_known_date + pd.Timedelta(days=1)

        prediction = int(
            np.round(MODEL.forecast(1)[0]).clip(0)
        )

        return jsonify({
            "api_code": "A15",
            "model_type": "SARIMA_Daily",
            "forecast_for_date": target_date.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


