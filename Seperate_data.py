import pandas as pd
df=pd.read_csv('API_Call_dataset.csv')
df.head()

top15=df['API Code'].value_counts().head(15)
print(top15)

for api_code in top15.index:
  filtered_df = df[df['API Code'] == api_code]
  filtered_df.to_csv(f"{api_code}.csv", index=False)