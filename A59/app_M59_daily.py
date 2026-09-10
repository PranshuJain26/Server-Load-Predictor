# http://127.0.0.1:5000/predict
import joblib
import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------


print("--- Loading M59 (Daily) Prophet Model ---")

MODEL_NAME = "A59\\M59_daily.joblib"
DATA_NAME = "A59\\A59_daily_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, parse_dates=["ds"])
    DATA = DATA.set_index("ds").asfreq("D")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT DAY call count for API A59 using Prophet."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.Timedelta(days=1)

        future = pd.DataFrame({"ds": [target_timestamp]})
        forecast = MODEL.predict(future)

        prediction = int(
            max(0, round(forecast["yhat"].iloc[0]))
        )

        return jsonify({
            "api_code": "A59",
            "model_type": "Prophet_Daily",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


