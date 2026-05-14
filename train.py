import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

melb_df = pd.read_csv('melb_data.csv', index_col=False)

# Feature selection
features = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'Lattitude', 'Longtitude']

# Predictor and Predicted
y = melb_df['Price']
X = melb_df[features]

# Split data into train and test
train_X, train_y, test_X, test_y = train_test_split(X, y, train_size=0.8, random_state=0)


# Function to compare different methods
def model_eval_mae(train_X, test_X, train_y, test_y):
    model = RandomForestRegressor(random_state=0)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    return mean_absolute_error(test_y,pred_y)