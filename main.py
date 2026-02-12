import os
import pandas as pd
import time

from ModelEvaluation.holdout_validation import holdout_validation
from ModelEvaluation.cross_validation import kfold_validation, find_optimal_k
from ModelEvaluation.stratified_shuffle_split_validation import stratified_shuffle_split_validation
from Preprocessing.feature_target_variables import load_data
from Preprocessing.data_cleaner import clean_data

def clear_screen():
    """
    Pulisce la schermata del terminale in modo portabile.
    """
    print("\n" * 100)

def run_holdout_validation(X, Y, k):
    """Esegue la validazione Holdout, richiedendo l'input finché non è valido."""
    while True:
        try:
            test_perc_str = input("Inserisci la percentuale per il test set (es. 0.2 per 20%): ")
            test_perc = float(test_perc_str)
            if not 0 < test_perc < 1:
                raise ValueError("La percentuale deve essere un numero compreso tra 0 e 1 (esclusi).")
            break
        except ValueError as e:
            print(f"Input non valido: {e}. Riprova.")
            time.sleep(2)

    train_size = int(len(X) * (1 - test_perc))
    if k >= train_size:
        print(f"Errore: Il numero di vicini (k={k}) non può essere >= alla dimensione del training set ({train_size}).")
        return

    holdout_validation(X, Y, k, test_perc)

def run_kfold_validation(X, Y, k):
    while True:
        try:
            K_folds_str = input("Inserisci il numero di fold (K) per la Cross Validation: ")
            K_folds = int(K_folds_str)
            if K_folds <= 1:
                raise ValueError("Il numero di fold deve essere maggiore di 1.")
            break
        except ValueError as e:
            print(f"Input non valido: {e}. Riprova.")
            time.sleep(2)

    # controllo se k= numero di vicini consultati dal KNN è minore della dimensione del training set in ogni fold
    train_size_per_fold = int(len(X) * (1 - 1/K_folds))
    if k >= train_size_per_fold:
        print(f"Errore: Il numero di vicini (k={k}) non può essere >="
              f" alla dimensione del training set in ogni fold ({train_size_per_fold}).")
        return

    kfold_validation(X, Y, k, K_folds)

def run_stratified_shuffle_split_validation(X, Y, k):
    while True:
        try:
            n_experiments = input("Inserisci il numero di Esperimenti per la Stratified shuffle split Validation: ")
            n_experiments = int(n_experiments)
            if n_experiments <= 1:
                raise ValueError("Il numero di Esperimenti deve essere maggiore di 1.")
            break
        except ValueError as e:
            print(f"Input non valido: {e}. Riprova.")
            time.sleep(2)
    # controllo se k= numero di vicini consultati dal KNN è minore della dimensione del training set in ogni esperimento
    # la proporzione di test è fissa al 20%
    train_size_per_experiment = int(len(X) * (1 - 0.2))
    if k >= train_size_per_experiment:
        print(f"Errore: Il numero di vicini (k={k}) non può essere >="
              f" alla dimensione del training set in ogni esperimento ({train_size_per_experiment}).")
        return
    stratified_shuffle_split_validation(X, Y, k, n_experiments)


def main():

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    # un po di stile per l'interfaccia
    welcome_message = "Benvenuto nel programma di classificazione k-NN"
    frame_line = "+" + "-" * (len(welcome_message) + 2) + "+"
    print(frame_line)
    print(f"| {welcome_message} |")
    print(frame_line)
    time.sleep(2)
    
    # Eseguiamo la pulizia dati
    # clean_data ora gestisce internamente la richiesta del file se non viene passato
    cleaned_path = None
    try:
        cleaned_path = clean_data()
    except KeyboardInterrupt:
        print("\nOperazione annullata dall'utente.")
        return
    except Exception as e:
        print(f"\nERRORE durante la pulizia dei dati: {e}")
        # Non ritorniamo qui, perché l'utente potrebbe voler caricare un file già pulito nel prossimo step

    # load_data ora gestisce il loop di richiesta file internamente se cleaned_path è None
    try:
        X, Y = load_data(cleaned_path)
    except KeyboardInterrupt:
        print("\nOperazione annullata dall'utente.")
        return
    except Exception as e:
        print(f"\nERRORE IRRECUPERABILE: {e}")
        return

    print(f"Dataset caricato: {len(X)} campioni con {len(X.columns)} feature.")
    time.sleep(2)
    input("\nPremi Invio per continuare al menu principale...")

    while True:
        clear_screen()
        print("="*50)
        print("MENU PRINCIPALE")
        print("="*50)
        print("1. Esegui validazione Holdout")
        print("2. Esegui K-Fold Cross Validation (Metodo B)")
        print("3. Esegui Stratified Shuffle Split (Metodo C)")
        print("4. Chiudi il programma ")
        print("="*50)

        try:
            choice_input = input("Inserisci la tua scelta (1-4): ")
            choice = int(choice_input)
        except ValueError:
            print("Scelta non valida. Inserisci un numero.")
            time.sleep(1)
            continue

        if choice not in [1, 2, 3, 4]:
            print("Scelta non valida. Riprova.")
            time.sleep(1)
            continue
            
        if choice == 4:
            clear_screen()
            print("Uscita dal programma. Arrivederci!")
            break

        while True:
            try:
                # sara chiesto il numero di vicini k per KNN in ogni caso, posso usare lo stesso input
                print("\nConfigurazione KNN:")
                print("="*50)
                print("Ricerca del valore k ottimale in corso...")
                optimal_k = find_optimal_k(X, Y)
                print(f"Il valore suggerito per k (basato su Error Rate) è: {optimal_k}")
                
                k_neighbors_str = input(f"Inserisci il numero di vicini (k) per KNN (invio per usare {optimal_k}): ").strip()
                
                if not k_neighbors_str:
                    k_neighbors = optimal_k
                else:
                    k_neighbors = int(k_neighbors_str)
                    
                if k_neighbors <= 0:
                    raise ValueError("Il numero di vicini deve essere un intero positivo.")
                break
            except ValueError as e:
                print(f"Input non valido: {e}. Riprova.")
                time.sleep(1)

        if choice == 1:
            run_holdout_validation(X, Y, k_neighbors)
        elif choice == 2:
            run_kfold_validation(X, Y, k_neighbors)
        elif choice == 3:
            run_stratified_shuffle_split_validation(X, Y, k_neighbors)

        another_run = input("\nVuoi eseguire un'altra operazione? (s/n): ").lower()

        if another_run != 's':
            clear_screen()
            print("Uscita dal programma. Arrivederci!")
            break

if __name__ == "__main__":
    main()
