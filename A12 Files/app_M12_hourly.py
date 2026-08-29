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


print("--- Loading M12 (Hourly) Prophet Model ---")

MODEL_NAME = "A12\\M12_hourly.joblib"
DATA_NAME = "A12\\A12_hourly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("H")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT HOUR call count for API A12 using Prophet."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Next hour timestamp
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.Timedelta(hours=1)

        # Prophet future dataframe
        future = pd.DataFrame({"ds": [target_timestamp]})
        forecast = MODEL.predict(future)

        prediction = int(
            np.round(forecast["yhat"].iloc[0]).clip(0)
        )

        return jsonify({
            "api_code": "A12",
            "model_type": "Prophet_Hourly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

