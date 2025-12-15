import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('../version_1.csv')

# 1. Uniforming the data: Replacing commas with dots and converting to numeric
cols_to_fix = ['Single Epithelial Cell Size', 'Bland Chromatin']

for col in cols_to_fix:
    try:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Error processing column '{col}': {e}")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 2. Investigating Missing Values and replacing invalid values
df_clean = df.copy()

import numpy as np

for column in df_clean.columns:
    try:
        # Create a temporary numeric version to find invalid values
        temp_numeric = pd.to_numeric(df_clean[column], errors='coerce')

        # Check if there are any invalid values (NaN, null, non-numeric)
        if temp_numeric.isnull().any():
            # Find mean of VALID numeric values and round up
            mean_value = temp_numeric.dropna().mean() if not temp_numeric.dropna().empty else 0
            mean_value_ceil = np.ceil(mean_value)

            # Replace only invalid values
            df_clean[column] = temp_numeric.fillna(mean_value_ceil)
            print(f"Replaced invalid values in column '{column}' with ceiling mean value: {mean_value_ceil}")
    except (ValueError, TypeError, IndexError) as e:
        print(f"Error processing column '{column}': {e}")
        continue

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()
print(f"Removed {len(df) - len(df_clean)} duplicate rows")

# Save the cleaned dataset
df_clean.to_csv('../version_1_cleaned.csv', index=False)
