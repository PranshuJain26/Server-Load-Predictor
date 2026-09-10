# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------


print("--- Loading M42 (Daily) Naive Model ---")

MODEL_NAME = "A42\\M42_daily.joblib"
DATA_NAME = "A42\\A42_daily_calls.csv"

try:
    ARTIFACT = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("D")

    print(f"Loaded model and data. Last date: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    ARTIFACT = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT DAY call count for API A42 using Naive model."""

    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known_day = DATA.index.max()
        target_day = last_known_day + pd.Timedelta(days=1)

        prediction = int(max(0, ARTIFACT["last_value"]))

        return jsonify({
            "api_code": "A42",
            "model_type": "Naive_Daily",
            "forecast_for_date": target_day.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

