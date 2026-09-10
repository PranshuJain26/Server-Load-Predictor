import pandas as pd
import warnings
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

print("--- Starting Final M2 (SARIMA, Daily) Training ---")

MODEL_NAME = "A2\\M2_daily.sarima"
DATA_SOURCE_RAW = "A2\\A2.csv"
DATA_SOURCE_PROCESSED = "A2\\A2_daily_calls.csv"

# --------------------------------------------------
# 1. Preprocessing
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")
try:
    df_raw = pd.read_csv(DATA_SOURCE_RAW)

    # Robust datetime parsing
    df_raw["datetime"] = pd.to_datetime(
        df_raw["Time of Call"],
        format="mixed",
        dayfirst=False
    )

    df_raw = df_raw.set_index("datetime")

    # Daily aggregation
    df_daily = df_raw.resample("D").size().to_frame("call_count")
    df_daily = df_daily.asfreq("D", fill_value=0)

    # Drop last incomplete day
    if not df_daily.empty:
        df_daily = df_daily.iloc[:-1]

    df_daily.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_daily)} days.")

except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --------------------------------------------------
# 2. Model Training (SARIMA)
# --------------------------------------------------
if len(df_daily) < 30:
    print("FATAL: Not enough daily data to train SARIMA.")
else:
    print("Training SARIMA model on full daily dataset...")

    model = SARIMAX(
        df_daily["call_count"],
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    print("Model training complete.")

    # --------------------------------------------------
    # 3. Save Model
    # --------------------------------------------------
    print(f"Saving model to '{MODEL_NAME}'...")
    joblib.dump(model, "A2\\M2_daily.joblib")
    print(f"Successfully saved '{MODEL_NAME}'.")

print("--- Daily Training Script Finished ---")
