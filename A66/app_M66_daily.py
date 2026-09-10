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


print("--- Loading M66 (Daily) XGBoost Model ---")

MODEL_NAME = "A66\\M66_daily.joblib"
DATA_NAME = "A66\\A66_daily_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("D")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT DAY call count for API A66 using XGBoost."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        next_index = np.array([[len(DATA)]])
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.Timedelta(days=1)

        prediction = int(
            max(0, round(MODEL.predict(next_index)[0]))
        )

        return jsonify({
            "api_code": "A66",
            "model_type": "XGBoost_Daily",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )


