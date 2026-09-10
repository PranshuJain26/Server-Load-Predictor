# http://127.0.0.1:5001/predict
import pandas as pd
import joblib
from flask import jsonify
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Initialize App
# --------------------------------------------------

print("--- Loading M2 (Daily) SARIMA Model and Data ---")

MODEL_NAME = "A2\\M2_daily.joblib"
DATA_NAME = "A2\\A2_daily_calls.csv"

try:
    MODEL = joblib.load(MODEL_NAME)
    print(f"✅ Loaded model: {MODEL_NAME}")

    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("D")
    print(f"Loaded data. Last date: {DATA.index.max()}")

except Exception as e:
    print(f"❌ Model load failed: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------

def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    last_known_date = DATA.index.max()
    target_timestamp = last_known_date + pd.Timedelta(days=1)

    forecast = MODEL.forecast(steps=1)[0]
    prediction = int(round(max(forecast, 0)))

    return jsonify({
        "api_code": "A2",
        "model_type": "SARIMA_Daily",
        "forecast_for_timestamp": target_timestamp.isoformat(),
        "predicted_call_count": prediction
    })

