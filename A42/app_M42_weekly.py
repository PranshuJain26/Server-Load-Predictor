# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------

print("--- Loading M42 (Weekly) Prophet Model ---")

MODEL_NAME = "A42\\M42_weekly.joblib"
DATA_NAME = "A42\\A42_weekly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME)
    DATA["ds"] = pd.to_datetime(DATA["ds"])

    print(f"Loaded model and data. Last week: {DATA['ds'].max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT WEEK call count for API A42 using Prophet."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_week = DATA["ds"].max()
        target_week = last_week + pd.Timedelta(weeks=1)

        future = pd.DataFrame({"ds": [target_week]})
        forecast = MODEL.predict(future)

        prediction = int(round(max(0, forecast["yhat"].iloc[0])))

        return jsonify({
            "api_code": "A42",
            "model_type": "Prophet_Weekly",
            "forecast_for_week": target_week.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

