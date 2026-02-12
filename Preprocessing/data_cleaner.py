import pandas as pd
import os

def clean_data(input_csv_path=None):
    """
    Pulisce il dataset specificato.
    Se input_csv_path non è fornito, chiede all'utente di inserirlo.
    """
    
    # Costruisce un percorso robusto per i file, partendo dalla posizione dello script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_container_dir = os.path.join(project_root, 'contenitore csv')

    while True:
        if input_csv_path is None:
            print(f"\n--- Pulizia Dati ---")
            
            # Elenca i file CSV disponibili in 'contenitore csv' per aiutare l'utente
            available_files = []
            if os.path.exists(csv_container_dir):
                available_files = [f for f in os.listdir(csv_container_dir) if f.lower().endswith('.csv')]
            
            if available_files:
                print(f"File CSV trovati in '{os.path.basename(csv_container_dir)}':")
                for f in available_files:
                    if not f.endswith('_cleaned.csv'): # Non mostrare i file già puliti
                        print(f"  - {f}")
            
            print("\n Posiziona i file nella cartella contenitore di csv oppure \n Inserisci il nome del file CSV da pulire (es. dati.csv) o il percorso relativo completo:")
            user_input = input().strip()
            
            if not user_input:
                print("ERRORE: Inserimento vuoto. Per favore inserisci un nome di file o un percorso.")
                continue
                
            # Rimuove eventuali virgolette
            user_input = user_input.strip('"').strip("'")
            
            # Se è solo un nome di file (senza directory), lo cerchiamo prima in 'contenitore csv'
            if not os.path.dirname(user_input):
                input_csv_path = os.path.join(csv_container_dir, user_input)
                # Se non esiste in 'contenitore csv', proviamo nella root
                if not os.path.exists(input_csv_path):
                    input_csv_path = os.path.join(project_root, user_input)
            else:
                # Se è un percorso completo o relativo con directory, lo usiamo direttamente
                input_csv_path = os.path.expanduser(user_input)
                input_csv_path = os.path.normpath(input_csv_path)

        if not input_csv_path.lower().endswith('.csv'):
            print(f"ERRORE: Il file '{input_csv_path}' non ha estensione .csv.")
            print("Per favore, seleziona un file CSV valido.\n")
            input_csv_path = None
            continue

        if not os.path.exists(input_csv_path):
            print(f"ERRORE: Il file '{input_csv_path}' non è stato trovato.")
            print("Assicurati di aver inserito il percorso corretto e riprova.\n")
            input_csv_path = None # Resetta per chiedere di nuovo
            continue

        # Percorso del file di output (verrà salvato nella stessa cartella del file di input)
        # Aggiungiamo '_cleaned' al nome del file originale
        input_dir = os.path.dirname(input_csv_path)
        input_filename = os.path.basename(input_csv_path)
        filename_without_ext = os.path.splitext(input_filename)[0]
        output_csv_path = os.path.join(input_dir, f"{filename_without_ext}_cleaned.csv")

        # Legge il file CSV in un DataFrame
        try:
            df = pd.read_csv(input_csv_path)
            print(f"File '{os.path.basename(input_csv_path)}' letto correttamente.")
            break # Esce dal ciclo while se la lettura ha successo
        except FileNotFoundError:
            # Questo catch è ridondante dato il check os.path.exists sopra, ma lo teniamo per sicurezza
            print(f"\nERRORE: Il file '{os.path.basename(input_csv_path)}' non è stato trovato.")
            input_csv_path = None
            continue
        except pd.errors.EmptyDataError:
            print(f"\nERRORE: Il file '{os.path.basename(input_csv_path)}' è vuoto e non può essere processato.")
            input_csv_path = None
            continue
        except Exception as e:
            print(f"\nERRORE inaspettato durante la lettura del file: {e}")
            input_csv_path = None
            continue

    # 1. Uniformare i dati: Sostituire le virgole con i punti e convertire in numerico
    cols_to_fix = ['Single Epithelial Cell Size', 'Bland Chromatin']

    for col in cols_to_fix:
        if col in df.columns:
            try:
                # Assicurarsi che la colonna sia di tipo stringa prima della sostituzione
                df[col] = df[col].astype(str).str.replace(',', '.')
                # Convertire in numerico, forzando gli errori a NaN (anche se ci aspettiamo che siano correggibili)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Errore durante l'elaborazione della colonna '{col}': {e}")
        else:
            print(f"Avviso: Colonna '{col}' non trovata nel dataset.")

    # Analisi dei valori mancanti e sostituzione dei valori non validi
    # Rimuovere le righe con valori target mancanti perché non possono essere utilizzate per l'apprendimento supervisionato
    if 'classtype_v1' in df.columns:
        df_clean = df.dropna(subset=['classtype_v1'])
    else:
        print("Avviso: Colonna target 'classtype_v1' non trovata. Impossibile rimuovere righe con target mancante.")
        df_clean = df.copy()


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
    try:
        df_clean.to_csv(output_csv_path, index=False)
        print(f"\nDataset pulito salvato in: {output_csv_path}")
        print("Dimensioni dopo la pulizia:", df_clean.shape)
        return output_csv_path
    except Exception as e:
        print(f"Errore durante il salvataggio del file pulito: {e}")
        return None
