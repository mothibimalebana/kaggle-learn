import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from model_ev import model_eval_mae


my_imputer = SimpleImputer()
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Read data
df = pd.read_csv('melb_data.csv')

# Define predictor and predictors
y = df.Price
X = df.drop(['Price'], axis='columns')

# Find columns with int or float dtype
num_cols = [col for col in X.columns if ((X[col].dtype == 'int') or X[col].dtype == 'float')]

# find columns with categorical values
low_cardinality_cols = [col for col in X.columns if ((X[col].dtype == 'str') and (X[col].nunique() <= 10))]


refined_cols = num_cols + low_cardinality_cols

X = df[refined_cols]

# List of predictors with missing values
missing_value_cols = [col for col in X.columns if X[col].isnull().any()]

X = X.drop(missing_value_cols, axis='columns')

# Split data into training vs test data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# Encoding categorical variables
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test_X[low_cardinality_cols]))

print(OH_cols_train)