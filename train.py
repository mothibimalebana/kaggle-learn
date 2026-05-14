import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from model_ev import model_eval_mae

melb_df = pd.read_csv('melb_data.csv', index_col=False)

# Feature selection
features = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'Lattitude', 'Longtitude', 'BuildingArea']

# Predictor and Predicted
y = melb_df['Price']
X = melb_df[features]

# Split data into train and test
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)

# Missing Values: Method 1
missing_value_columns = [col for col in train_X.columns if train_X[col].isnull().any()]

train_X = train_X.drop(missing_value_columns, axis=1)
test_X = test_X.drop(missing_value_columns, axis=1)

model_mae = model_eval_mae(train_X, test_X, train_y, test_y)
print(model_mae)