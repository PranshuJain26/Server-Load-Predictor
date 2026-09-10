# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------

print("--- Loading M45 (Weekly) Naive Model ---")

MODEL_NAME = "A45\\M45_weekly.joblib"
DATA_NAME = "A45\\A45_weekly_calls.csv"

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
    """Predict NEXT WEEK call count for API A45 using Naive model."""

    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_week = DATA.index.max()
        target_week = last_week + pd.Timedelta(weeks=1)

        prediction = int(max(0, ARTIFACT["last_value"]))

        return jsonify({
            "api_code": "A45",
            "model_type": "Naive_Weekly",
            "forecast_for_week": target_week.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

