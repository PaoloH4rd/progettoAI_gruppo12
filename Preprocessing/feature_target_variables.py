import pandas as pd

# Carica il dataset pulito
df_clean = pd.read_csv('../version_1_cleaned.csv')

# Definisce la variabile target
target_col = 'classtype_v1'

# Definisce le feature (tutte le colonne tranne la variabile target)
feature_cols = [col for col in df_clean.columns if col not in [target_col]]

# Crea il dataset delle feature (X) e la variabile target (y)
X = df_clean[feature_cols]
y = df_clean[target_col]
