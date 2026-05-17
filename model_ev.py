from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Function to compare different methods
def model_eval_mae(train_X, test_X, train_y, test_y):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return mean_absolute_error(test_y, preds)