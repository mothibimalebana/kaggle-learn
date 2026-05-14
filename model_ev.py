from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Function to compare different methods
def model_eval_mae(train_X, test_X, train_y, test_y):
    """
    input: train_X, test_X, train_y, test_y
    output: mean_absolute_error
    """
    model = RandomForestRegressor(random_state=0)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    return mean_absolute_error(test_y,pred_y)