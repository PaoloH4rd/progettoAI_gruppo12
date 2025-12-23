import pandas as pd
import os

# Ottiene il percorso della directory padre di questo file
current_dir = os.path.dirname(os.path.abspath(__file__))
# risalgo alla cartella principale del progetto
project_root = os.path.dirname(current_dir)
os.chdir(project_root)
cleaned_file_path = os.path.join(project_root, 'version_1_cleaned.csv')

# Carica il dataset pulito
df_clean = pd.read_csv(cleaned_file_path)

# Definisce la variabile target
target_col = 'classtype_v1'

# Definisce le feature (tutte le colonne tranne la variabile target)
feature_cols = [col for col in df_clean.columns if col not in [target_col]]

# Crea il dataset delle feature (X) e la variabile target (Y)
X = df_clean[feature_cols]
Y = df_clean[target_col]
