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

print("--- Loading M10 (Hourly) XGBoost Model ---")

MODEL_NAME = "A10\\M10_hourly.joblib"
DATA_NAME = "A10\\A10_hourly_calls.csv"

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
    """Predict NEXT HOUR call count for API A10 using XGBoost."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Next hour timestamp
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.Timedelta(hours=1)

        # Time index feature (same as training)
        X_next = np.array([[len(DATA)]])

        prediction = int(
            np.round(MODEL.predict(X_next)[0]).clip(0)
        )

        return jsonify({
            "api_code": "A10",
            "model_type": "XGBoost_Hourly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

