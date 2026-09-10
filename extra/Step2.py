import pandas as pd
df = pd.read_csv("final_dataset_each_api_all_apps_random.csv")
df['Time of Call'] = pd.to_datetime(df['Time of Call'])

hourly_all = (
    df
    .groupby(['API Code', 'App Name'])
    .resample('h', on='Time of Call')
    .size()
    .reset_index(name='call_count')
)

hourly_all.to_csv("all_apis_hourly_with_app.csv", index=False)

print("✅ all_apis_hourly_with_app.csv created")