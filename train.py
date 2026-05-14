import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from model_ev import model_eval_mae
from sklearn.impute import SimpleImputer


melb_df = pd.read_csv('melb_data.csv', index_col=False)
my_imputer = SimpleImputer()
# Feature selection
features = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'Lattitude', 'Longtitude', 'BuildingArea']

# Predictor and Predicted
y = melb_df['Price']
X = melb_df[features]

# Split data into train and test
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)

# Missing Values: Method 1
missing_value_columns = [col for col in train_X.columns if train_X[col].isnull().any()]

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_test_X = pd.DataFrame(my_imputer.fit_transform(test_X))

imputed_train_X.columns = train_X.columns
imputed_test_X.columns = test_X.columns
print(imputed_train_X)
print(imputed_test_X)

model_mae = model_eval_mae(imputed_train_X, imputed_test_X, train_y, test_y)
print(model_mae)