import pandas as pd

winemag_df = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)

print(winemag_df.iloc[:3, 0])