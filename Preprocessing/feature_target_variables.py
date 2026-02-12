import pandas as pd
import os

def load_data(cleaned_file_path=None):
    """
    Carica il dataset pulito e restituisce le feature (X) e la variabile target (Y).
    Se cleaned_file_path non è fornito, lo chiede all'utente.
    """
    # Ottiene il percorso della directory padre di questo file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Risale alla cartella principale del progetto
    project_root = os.path.dirname(current_dir)
    csv_container_dir = os.path.join(project_root, 'contenitore csv')

    while True:
        if cleaned_file_path is None:
            print(f"\n--- Caricamento Dati ---")
            
            # Elenca i file puliti disponibili
            available_cleaned = []
            
            # Cerca nella root
            available_cleaned += [os.path.join(project_root, f) for f in os.listdir(project_root) if f.endswith('_cleaned.csv')]
            # Cerca in 'contenitore csv'
            if os.path.exists(csv_container_dir):
                available_cleaned += [os.path.join(csv_container_dir, f) for f in os.listdir(csv_container_dir) if f.endswith('_cleaned.csv')]

            if available_cleaned:
                print("File puliti trovati:")
                for i, path in enumerate(available_cleaned):
                    print(f"  {i+1}. {os.path.basename(path)}")
                
                print("\nInserisci il numero del file, il nome (es. version_1_cleaned.csv) o il percorso completo:")
                user_input = input().strip()
                
                if not user_input:
                    print("ERRORE: Inserimento vuoto.")
                    continue
                
                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(available_cleaned):
                        cleaned_file_path = available_cleaned[idx]
                    else:
                        print(f"ERRORE: Indice {idx+1} non valido.")
                        continue
                else:
                    # Rimuove eventuali virgolette
                    user_input = user_input.strip('"').strip("'")
                    
                    if not os.path.dirname(user_input):
                        # Cerca prima in 'contenitore csv'
                        potential_path = os.path.join(csv_container_dir, user_input)
                        if os.path.exists(potential_path):
                            cleaned_file_path = potential_path
                        else:
                            # Prova nella root
                            cleaned_file_path = os.path.join(project_root, user_input)
                    else:
                        cleaned_file_path = os.path.expanduser(user_input)
                        cleaned_file_path = os.path.normpath(cleaned_file_path)
            else:
                print("Nessun file '_cleaned.csv' trovato automaticamente.")
                print("Per favore, inserisci il percorso completo del file pulito:")
                user_input = input().strip()
                if not user_input:
                    continue
                cleaned_file_path = os.path.normpath(os.path.expanduser(user_input.strip('"').strip("'")))

        if not os.path.exists(cleaned_file_path):
            print(f"ERRORE: Il file '{cleaned_file_path}' non è stato trovato.")
            cleaned_file_path = None
            continue
        
        break

    # Carica il dataset pulito
    df_clean = pd.read_csv(cleaned_file_path)
    print(f"File '{os.path.basename(cleaned_file_path)}' caricato con successo.")

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
