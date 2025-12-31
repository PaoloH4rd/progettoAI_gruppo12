import pandas as pd
import os

def load_data():
    """
    Carica il dataset pulito e restituisce le feature (X) e la variabile target (Y).
    """
    # Ottiene il percorso della directory padre di questo file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Risale alla cartella principale del progetto
    project_root = os.path.dirname(current_dir)
    # Costruisce un percorso assoluto per il file pulito
    cleaned_file_path = os.path.join(project_root, 'version_1_cleaned.csv')

    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"Il file {cleaned_file_path} non è stato trovato. Assicurati di aver eseguito prima lo script di pulizia dei dati.")

    # Carica il dataset pulito
    df_clean = pd.read_csv(cleaned_file_path)

    # Definisce la variabile target
    target_col = 'classtype_v1'

    Y = df_clean[target_col]
    classes = pd.unique(Y)
    
    # Verifica che ci siano esattamente le due classi attese (2 e 4)
    if set(classes) != {2, 4}:
        raise ValueError(f"Il dataset deve contenere esattamente le classi 2 e 4. Trovate: {classes}")
    
    # Conversione a monte: 2 (Benigno) -> 0, 4 (Maligno) -> 1
    # Convertiamo prima in int per sicurezza (nel caso siano float 2.0/4.0) e poi mappiamo
    df_clean[target_col] = df_clean[target_col].astype(int).map({2: 0, 4: 1})

    # Definisce le feature (tutte le colonne tranne la variabile target)
    feature_cols = [col for col in df_clean.columns if col not in [target_col]]
    
    # Crea il dataset delle feature (X) e la variabile target (Y)
    X = df_clean[feature_cols]
    Y = df_clean[target_col]

    return X, Y
