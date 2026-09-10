# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------


print("--- Loading M59 (Weekly) Naive Model ---")

MODEL_NAME = "A59\\M59_weekly.joblib"
DATA_NAME = "A59\\A59_weekly_calls.csv"

try:
    LAST_VALUE = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, parse_dates=["ds"])
    DATA = DATA.set_index("ds").asfreq("W")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    LAST_VALUE = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT WEEK call count for API A59 using Naive model."""

    if LAST_VALUE is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.Timedelta(weeks=1)

        prediction = int(max(0, round(LAST_VALUE)))

        return jsonify({
            "api_code": "A59",
            "model_type": "Naive_Weekly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


