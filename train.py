import pandas as pd

# read data and store data in Dataframe titled melbourne_data
melbourne_data = pd.read_csv('melb_data.csv')

# print summary of data 
melbourne_data.describe()

print(melbourne_data.describe())