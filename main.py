import os
import pandas as pd
import time
from ModelEvaluation.cross_validation import kfold_validation
from ModelEvaluation.holdout_validation import holdout_validation
from Preprocessing.feature_target_variables import load_data
from Preprocessing.data_cleaner import clean_data


def clear_screen():
    """
    Pulisce la schermata del terminale in modo portabile.
    Stampa 100 righe vuote per simulare la pulizia dello schermo,
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

    print("\n" + "="*60)
    print("AVVISO: I risultati dettagliati e i grafici sono stati salvati.")
    print("Controlla la cartella 'output' nella directory del progetto.")
    print("="*60)

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

    train_size_per_fold = int(len(X) * (1 - 1/K_folds))
    if k >= train_size_per_fold:
        print(f"Errore: Il numero di vicini (k={k}) non può essere >= alla dimensione del training set in ogni fold ({train_size_per_fold}).")
        return

    kfold_validation(X, Y, k, K_folds)

    print("\n" + "="*60)
    print("AVVISO: I risultati dettagliati e i grafici sono stati salvati.")
    print("Controlla la cartella 'output' nella directory del progetto.")
    print("="*60)


def main():
    if not os.path.exists('version_1.csv'):
        print("ERRORE: File 'version_1.csv' non trovato. Assicurati che sia nella root del progetto.")
        input("Premi Invio per uscire.")
        return

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    # un po di stile per l'interfaccia
    welcome_message = "Benvenuto nel programma di classificazione k-NN"
    frame_line = "+" + "-" * (len(welcome_message) + 2) + "+"
    print(frame_line)
    print(f"| {welcome_message} |")
    print(frame_line)
    time.sleep(2)
    clean_data()
    X, Y = load_data()
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
        print("="*50)

        try:
            choice = int(input("Inserisci la tua scelta (1-3): "))
        except ValueError:
            print("Scelta non valida. Inserisci un numero.")
            time.sleep(1)
            continue

        if choice not in [1, 2, 3]:
            print("Scelta non valida. Riprova.")
            time.sleep(1)
            continue

        while True:
            try:
                k_neighbors_str = input("\nInserisci il numero di vicini (k) per KNN: ")
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
            pass
        another_run = input("\nVuoi eseguire un'altra operazione? (s/n): ").lower()
        if another_run != 's':
            clear_screen()
            print("Uscita dal programma. Arrivederci!")
            break

if __name__ == "__main__":
    main()
