import pandas as pd
from sklearn.model_selection import train_test_split 

# read data and store data in Dataframe titled melbourne_data
melbourne_data = pd.read_csv('melb_data.csv')

melbourne_features = ['Suburb', 'Rooms', 'Type', 'Landsize']
print(melbourne_data.columns)

y = melbourne_data.Price
X = melbourne_data[melbourne_features] 

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

print("="*30+"train"+"="*30)
print(train_X)

print("="*30+"test"+"="*30)
print(train_y)
