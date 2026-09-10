# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------


print("--- Loading M40 (Daily) Prophet Model ---")

MODEL_NAME = "A40\\M40_daily.joblib"
DATA_NAME = "A40\\A40_daily_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME)
    DATA["ds"] = pd.to_datetime(DATA["ds"])

    print(f"Loaded model and data. Last date: {DATA['ds'].max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT DAY call count for API A40 using Prophet."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_date = DATA["ds"].max()
        target_date = last_date + pd.Timedelta(days=1)

        future = pd.DataFrame({"ds": [target_date]})
        forecast = MODEL.predict(future)

        prediction = int(round(max(0, forecast["yhat"].iloc[0])))

        return jsonify({
            "api_code": "A40",
            "model_type": "Prophet_Daily",
            "forecast_for_date": target_date.date().isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

