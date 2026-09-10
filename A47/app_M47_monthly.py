import joblib
import pandas as pd
from flask import jsonify
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App and Load Artifacts
# --------------------------------------------------

print("--- Loading M47 (Monthly) Random Forest Model ---")

MODEL_NAME = "A47\\M47_monthly.joblib"
DATA_NAME = "A47\\A47_monthly_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("M")

    print(f"Loaded model and data. Last timestamp: {DATA.index.max()}")

except Exception as e:
    print(f"FATAL: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------
def predict():
    """Predict NEXT MONTH call count for API A47 using Random Forest."""

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        last_known = DATA.index.max()
        target_timestamp = last_known + pd.offsets.MonthEnd(1)

        next_index = np.array([[len(DATA)]])

        prediction = int(
            max(0, round(MODEL.predict(next_index)[0]))
        )

        return jsonify({
            "api_code": "A47",
            "model_type": "RandomForest_Monthly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": prediction
        })

    except Exception as e:
        return jsonify(
            {"error": f"Prediction failed: {e}"},
            500
        )

