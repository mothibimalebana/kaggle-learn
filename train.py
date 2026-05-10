import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

# read data and store data in Dataframe titled melbourne_data
melbourne_data = pd.read_csv('melb_data.csv')

# feature selected
melbourne_features = ['Suburb', 'Rooms', 'Type', 'Landsize']

# variables
y = melbourne_data.Price
X = melbourne_data[melbourne_features]

# split data into test --> 20% and train --> 80%
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

melbourne_model = RandomForestRegressor(random_state=1)

melbourne_model.fit(train_X, train_y)

