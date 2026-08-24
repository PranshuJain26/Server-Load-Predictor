import pandas as pd

# Load CSV
df = pd.read_csv(r"A10\A10.csv")
print("CSV loaded")

# Convert timestamp
df["Time of Call"] = pd.to_datetime(df["Time of Call"])

# Each row = 1 API call
df["call_count"] = 1

# Set index
df.set_index("Time of Call", inplace=True)

# Safe aggregation functions
def safe_mode(series):
    if series.empty:
        return None
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return None

agg_rules = {
    "API Code": "first",
    "App Name": safe_mode,
    "call_count": "sum"
}

# Resample & save
df.resample("h").agg(agg_rules).to_csv(r"A10\A10_hourly.csv")
df.resample("D").agg(agg_rules).to_csv(r"A10\A10_daily.csv")
df.resample("W").agg(agg_rules).to_csv(r"A10\A10_weekly.csv")
df.resample("M").agg(agg_rules).to_csv(r"A10\A10_monthly.csv")

print("✅ A10 data prepared with all columns (safe)")
