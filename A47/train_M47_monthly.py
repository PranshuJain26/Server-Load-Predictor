import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("--- Starting M47 (Random Forest, Monthly) Training ---")

MODEL_NAME = "A47\\M47_monthly.joblib"
DATA_SOURCE_RAW = "A47\\A47.csv"
DATA_SOURCE_PROCESSED = "A47\\A47_monthly_calls.csv"

# --------------------------------------------------
# 1. Preprocessing (MONTHLY)
# --------------------------------------------------
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")

df_raw = pd.read_csv(DATA_SOURCE_RAW)

df_raw["datetime"] = pd.to_datetime(
    df_raw["Time of Call"],
    format="mixed",
    dayfirst=False
)

df_raw = df_raw.set_index("datetime")

df_monthly = df_raw.resample("M").size().to_frame("call_count")
df_monthly = df_monthly.asfreq("M", fill_value=0)

# Drop last incomplete month
if not df_monthly.empty:
    df_monthly = df_monthly.iloc[:-1]

df_monthly.to_csv(DATA_SOURCE_PROCESSED)
print(f"Saved monthly data → {DATA_SOURCE_PROCESSED} ({len(df_monthly)} rows)")

# --------------------------------------------------
# 2. Train / Validation Split (80%)
# --------------------------------------------------
y = df_monthly["call_count"].values
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
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Validation RMSE: {rmse:.3f}")

# --------------------------------------------------
# 5. Save Model
# --------------------------------------------------
joblib.dump(model, MODEL_NAME)
print(f"Saved model → {MODEL_NAME}")

print("--- M47 Monthly Random Forest Training Finished ---")
