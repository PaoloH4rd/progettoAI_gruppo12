import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('../version_1.csv')

# Print initial info about the DataFrame
print(df.info())

# 1. Uniforming the data: Replacing commas with dots and converting to numeric
cols_to_fix = ['Single Epithelial Cell Size', 'Bland Chromatin']

for col in cols_to_fix:
    try:
        # Be sure the column is string type before replacement
        df[col] = df[col].astype(str).str.replace(',', '.')
        # Convert to numeric, forcing errors to NaN (even if we expect them to be fixable)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Error processing column '{col}': {e}")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 2. Investigating Missing Values and replacing invalid values
# Verify the count of null values in each column

null_counts = df.isnull().sum()
print("Conteggio valori nulli per colonna:\n", null_counts)

# Check rows where the target variable 'classtype_v1' is missing
missing_target_df = df[df['classtype_v1'].isnull()]
print("\nPrime 5 righe con 'classtype_v1' mancante:")
print(missing_target_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Strategy: Remove rows with missing target values because they cannot be used for supervised learning
df_clean = df.dropna(subset=['classtype_v1'])

# Check null values again after removing rows with missing target
print("\nConteggio valori nulli dopo aver eliminato i target mancanti:\n", df_clean.isnull().sum())

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()
print(f"Removed {len(df) - len(df_clean)} duplicate rows")

# Remove columns that are not useful for analysis
columns_to_drop = ['Sample code number', 'BareNucleix_wrong', 'Blood Pressure', 'Heart Rate']
df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

# Save the cleaned dataset
df_clean.to_csv('../version_1_cleaned.csv', index=False)

print("\nDimensioni dopo la pulizia:", df_clean.shape)

df_clean= pd.read_csv('../version_1_cleaned.csv')


