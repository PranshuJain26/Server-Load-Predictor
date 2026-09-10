import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M69 (RandomForest, Daily) Training ---")

MODEL_NAME = "A69\\M69_daily.joblib"
DATA_SOURCE_RAW = "A69\\A69.csv"
DATA_SOURCE_PROCESSED = "A69\\A69_daily_calls.csv"

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

df_daily = (
    df_raw
    .groupby(pd.Grouper(key="datetime", freq="D"))
    .size()
    .to_frame("call_count")
)

df_daily = df_daily.asfreq("D", fill_value=0)

# Drop last incomplete day
if len(df_daily) > 1:
    df_daily = df_daily.iloc[:-1]

df_daily.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved daily data → {DATA_SOURCE_PROCESSED} ({len(df_daily)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_daily["call_count"].values
split = int(len(y) * 0.8)

y_train, y_test = y[:split], y[split:]

X_train = np.arange(len(y_train)).reshape(-1, 1)
X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

# --------------------------------------------------
# 3. Train Random Forest Model
# --------------------------------------------------
print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Validation RMSE
# --------------------------------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M69 Daily Random Forest Training Finished ---")
