import pandas as pd
import os

def clean_data():
    # Costruisce un percorso robusto per i file, partendo dalla posizione dello script.
    # __file__ è il percorso dello script corrente (data_cleaner.py)
    # os.path.dirname() ottiene la directory di quel file ('.../Preprocessing')
    # Un secondo os.path.dirname() risale di un livello alla root del progetto ('.../progettoAI_gruppo12')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Percorso del file di input (nella root del progetto)
    input_csv_path = os.path.join(project_root, 'version_1.csv')

    # Percorso del file di output (verrà salvato nella root del progetto)
    output_csv_path = os.path.join(project_root, 'version_1_cleaned.csv')

    # Legge il file CSV in un DataFrame
    try:
        df = pd.read_csv(input_csv_path)
        print(f"File '{os.path.basename(input_csv_path)}' letto correttamente.")
    except FileNotFoundError:
        print(f"\nERRORE: Il file '{os.path.basename(input_csv_path)}' non è stato trovato nella directory principale del progetto.")
        print("Per favore, assicurati che il file sia presente e riprova.")
        return  # Interrompe l'esecuzione se il file non può essere letto
    except pd.errors.EmptyDataError:
        print(f"\nERRORE: Il file '{os.path.basename(input_csv_path)}' è vuoto e non può essere processato.")
        return
    except Exception as e:
        print(f"\nERRORE inaspettato durante la lettura del file: {e}")
        return

    # 1. Uniformare i dati: Sostituire le virgole con i punti e convertire in numerico
    cols_to_fix = ['Single Epithelial Cell Size', 'Bland Chromatin']

    for col in cols_to_fix:
        try:
            # Assicurarsi che la colonna sia di tipo stringa prima della sostituzione
            df[col] = df[col].astype(str).str.replace(',', '.')
            # Convertire in numerico, forzando gli errori a NaN (anche se ci aspettiamo che siano correggibili)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Errore durante l'elaborazione della colonna '{col}': {e}")

    # Analisi dei valori mancanti e sostituzione dei valori non validi
    # Rimuovere le righe con valori target mancanti perché non possono essere utilizzate per l'apprendimento supervisionato
    df_clean = df.dropna(subset=['classtype_v1'])


    # Imputazione dei valori mancanti con la media
    # Per ogni colonna che ha ancora valori mancanti, sostituiamo i NaN con la media di quella colonna.
    # Questo è un approccio comune per gestire i dati mancanti senza perdere righe intere.
    print("\nInizio imputazione dei valori mancanti con la media...")
    imputation_performed = False
    for col in df_clean.columns:
        # Applica l'imputazione solo a colonne numeriche che hanno valori mancanti
        if pd.api.types.is_numeric_dtype(df_clean[col]) and df_clean[col].isnull().any():
            # Calcola la media della colonna (ignorando i valori NaN esistenti)
            mean_value = df_clean[col].mean()
            # Sostituisci i NaN con la media calcolata
            df_clean[col].fillna(mean_value, inplace=True)
            print(f"  - Imputati valori mancanti nella colonna '{col}' con la media ({mean_value:.2f}).")
            imputation_performed = True

    if not imputation_performed:
        print("  - Nessuna imputazione necessaria, non ci sono valori mancanti nelle colonne numeriche.")

    # Rimozione delle righe duplicate
    # Questo passaggio viene eseguito dopo la pulizia e l'imputazione per massimizzare l'efficacia,
    # catturando duplicati che potevano essere mascherati da valori mancanti o formattazione errata.
    print("\nRimozione delle righe duplicate...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"  - Rimosse {initial_rows - len(df_clean)} righe duplicate.")


    # Rimuovere le colonne che non sono utili per l'analisi
    columns_to_drop = ['Sample code number', 'BareNucleix_wrong', 'Blood Pressure', 'Heart Rate']
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

    # Salvare il dataset pulito
    df_clean.to_csv(output_csv_path, index=False)

    print("\nDimensioni dopo la pulizia:", df_clean.shape)
