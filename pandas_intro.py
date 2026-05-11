import pandas as pd

df = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)

review_points_mean = df.points.mean()

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

df_remean = df.apply(remean_points, axis='columns')
print(df_remean)