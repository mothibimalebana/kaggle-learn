import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


my_imputer = SimpleImputer()

# Read data
df = pd.read_csv('melb_data.csv')

# Find columns with int or float dtype
num_cols = [col for col in df.columns if ((df[col].dtype == 'int') or df[col].dtype == 'float')]

# find columns with categorical values
categorical_cols = [col for col in df.columns if ((df[col].dtype == 'str') and (df[col].nunique() <= 10))]


refined_cols = num_cols + categorical_cols

df_num_cols = df[refined_cols]

# Define predictor and predictors
y = df.Price
X = df_num_cols.drop(['Price'], axis='columns')

# List of predictors with missing values
missing_value_cols = [col for col in X.columns if X[col].isnull().any()]
print(missing_value_cols)
X = X.drop(missing_value_cols, axis='columns')


# Split data into training vs test data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# For simplicity we drop missing columns
print(train_X[categorical_cols])
