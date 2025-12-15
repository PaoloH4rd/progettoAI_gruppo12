import pandas as pd

# Load the cleaned dataset
df_clean = pd.read_csv('../version_1_cleaned.csv')

# Define features and target variable
target_col = 'classtype_v1'

feature_cols = [col for col in df_clean.columns if col not in [target_col]]

X = df_clean[feature_cols]
y = df_clean[target_col]