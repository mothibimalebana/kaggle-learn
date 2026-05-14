import pandas as pd

melb_df = pd.read_csv('melb_data.csv', index_col=False)

print(melb_df.head())