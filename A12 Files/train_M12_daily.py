import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M12 (RandomForest, Daily) Training ---")

MODEL_NAME = "A12\\M12_daily.joblib"
DATA_SOURCE_RAW = "A12\\A12.csv"
DATA_SOURCE_PROCESSED = "A12\\A12_daily_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (DAILY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

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
print(f"Saved daily data → {DATA_SOURCE_PROCESSED} ({len(df_daily)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_daily["call_count"].values

split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Time Index Feature (UNCHANGED LOGIC)
# --------------------------------------------------
X_train = np.arange(len(y_train)).reshape(-1, 1)
X_test = np.arange(
    len(y_train),
    len(y_train) + len(y_test)
).reshape(-1, 1)

# --------------------------------------------------
# 4. Train RandomForest Model
# --------------------------------------------------
print("Training RandomForest model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Validation RMSE
# --------------------------------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 6. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M12 Daily RandomForest Training Finished ---")
