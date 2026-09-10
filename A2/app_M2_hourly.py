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
print("--- Loading M2 (Hourly) XGBoost Model and Data ---")

MODEL_NAME = "A2\\M2_hourly.joblib"
DATA_NAME = "A2\\A2_hourly_calls.csv"

try:
    # Load trained XGBoost model
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    # Load historical hourly data (for feature calculation)
    DATA = pd.read_csv(DATA_NAME, index_col="datetime", parse_dates=True)
    DATA = DATA.asfreq("H")
    print(f"Successfully loaded {DATA_NAME}. Last data point: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Run train_M2_hourly.py first. Error: {e}")
    MODEL = None

# --------------------------------------------------
# 2. Prediction Endpoint
# --------------------------------------------------

def predict():
    """Predicts the NEXT HOUR call count for API A2 using XGBoost."""

    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        # 1. Target timestamp (next hour)
        last_known_date = DATA.index.max()
        target_timestamp = last_known_date + pd.Timedelta(hours=1)

        print(f"Prediction requested for: {target_timestamp.isoformat()}")

        # 2. Required timestamps for features
        lag_24_timestamp = target_timestamp - pd.Timedelta(hours=24)
        rolling_window_start = target_timestamp - pd.Timedelta(hours=24)
        rolling_window_end = target_timestamp - pd.Timedelta(hours=1)

        # 3. Validate historical data availability
        if lag_24_timestamp not in DATA.index:
            return jsonify({
                "error": f"Not enough historical data. Need data from {lag_24_timestamp}."
            }), 400

        if rolling_window_start not in DATA.index:
            return jsonify({
                "error": f"Not enough historical data. Need data from {rolling_window_start}."
            }), 400

        # 4. Build feature vector (MUST MATCH TRAINING ORDER)
        features = []

        # Time-based features
        features.append(target_timestamp.hour)
        features.append(target_timestamp.dayofweek)
        features.append(target_timestamp.month)

        # Lag feature
        lag_24_value = DATA.loc[lag_24_timestamp]["call_count"]
        features.append(lag_24_value)

        # Rolling mean feature
        rolling_mean_24_value = DATA.loc[
            rolling_window_start:rolling_window_end
        ]["call_count"].mean()
        features.append(rolling_mean_24_value)

        # 5. Predict
        feature_vector = np.array(features).reshape(1, -1)
        prediction_raw = MODEL.predict(feature_vector)

        final_prediction = int(np.round(prediction_raw[0]).clip(0))

        # 6. Prepare response
        features_dict = {
            "hour": int(features[0]),
            "dayofweek": int(features[1]),
            "month": int(features[2]),
            "lag_24": int(features[3]),
            "rolling_mean_24": float(features[4])
        }

        return jsonify({
            "api_code": "A2",
            "model_type": "XGBoost_Hourly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction,
            "features_used": features_dict
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify(
            {"error": f"An error occurred during prediction: {e}"},
            500
        )


