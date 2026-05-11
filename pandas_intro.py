import pandas as pd

df = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)

review_points_mean = df.points
print(review_points_mean)

df.points.map(lam)