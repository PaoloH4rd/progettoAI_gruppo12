import time
import random

from ModelDevelopment.knn_scratch import KNN
from ModelEvaluation.results_handler import KFoldResultsHandler
from .metrics import calculate_metrics


def k_fold_split(X, Y, k_folds=5):
    """
    Suddivide i dati in k fold per la K-Fold Cross Validation standard.
    Ogni fold viene utilizzato una volta come set di test, mentre i restanti k-1 fold
    vengono usati come set di addestramento.

    Args:
        X (list): Lista di feature.
        Y (list): Lista di label.
        k_folds (int): Numero di fold.

    Returns:
        list: Una lista di tuple. Ogni tupla rappresenta un fold e contiene
              (X_train, Y_train, X_test, Y_test).
    """
    # 1. Creazione e mescolamento degli indici
    # Crea una lista di indici da 0 alla lunghezza del dataset.
    indices = list(range(len(X)))
    # Mescola gli indici in modo casuale. Questo assicura che la suddivisione
    # non sia influenzata dall'ordine originale dei dati.
    random.shuffle(indices)

    # 2. Calcolo della dimensione dei fold
    # Calcola la dimensione approssimativa di ogni fold.
    fold_size = len(X) // k_folds
    # Lista per contenere i dati di tutti i fold.
    folds = []

    # 3. Creazione di ogni fold
    # Itera k volte, una per ogni fold da creare.
    for i in range(k_folds):
        # 4. Definizione degli indici di test per il fold corrente
        # Calcola l'indice di inizio del blocco di test.
        test_start = i * fold_size
        # Calcola l'indice di fine. L'ultimo fold prende tutti gli indici rimanenti
        # per gestire i casi in cui la dimensione del dataset non è perfettamente divisibile per k.
        test_end = test_start + fold_size if i < k_folds - 1 else len(X)
        # Estrae gli indici per il set di test da quelli mescolati.
        test_indices = indices[test_start:test_end]

        # 5. Definizione degli indici di training
        # Gli indici di training sono tutti quelli che non sono nel set di test.
        # Concateniamo tutto ciò che sta prima dell'inizio del test con tutto ciò che sta dopo la fine del test.
        train_indices = indices[:test_start] + indices[test_end:]

        # 6. Creazione dei set di dati
        # Usa gli indici per creare i set di dati di training e test.
        X_train = [X[j] for j in train_indices]
        Y_train = [Y[j] for j in train_indices]
        X_test = [X[j] for j in test_indices]
        Y_test = [Y[j] for j in test_indices]

        # 7. Aggiunta del fold alla lista
        # Aggiunge la tupla con i dati del fold corrente alla lista dei folds.
        folds.append((X_train, Y_train, X_test, Y_test))

    # 8. Restituzione dei fold
    return folds


def evaluate_kfold(X, Y, knn_model_class, k_neighbors, k_folds=5):
    """
    Esegue una validazione K-Fold sull'intero dataset.
    1. Suddivide l'INTERO dataset in K parti (fold).
    2. Per ogni iterazione (fold), usa 1 parte come Test Set e le restanti K-1 come Training Set.
    3. Calcola le metriche per ognuno dei K esperimenti e le restituisce.
    """
    # 1. PREPARAZIONE PER LA K-FOLD CROSS VALIDATION
    # Suddivide l'intero dataset (X, Y) in 'k' fold.
    # Questo assicura che ogni singolo esempio del dataset venga usato esattamente una volta per il test.
    folds = k_fold_split(X, Y, k_folds)
    all_fold_metrics = []
    all_fold_raw_data = []

    print(f"\n{'=' * 60}")
    print(f"INIZIO K-FOLD CROSS VALIDATION (k={k_folds})")
    print(f"Totale campioni nel dataset: {len(X)}")
    print(f"{'=' * 60}\n")

    # 2. ESECUZIONE DELLA K-FOLD CROSS VALIDATION
    # Itera su ogni fold. A ogni iterazione, un fold diverso viene usato come test set
    # e i restanti k-1 fold vengono usati come training set.
    for fold_num, (X_train_fold, Y_train_fold, X_test_fold, Y_test_fold) in enumerate(folds, 1):
        print(f"  - Esperimento {fold_num}/{k_folds}")
        print(f"    Training samples: {len(X_train_fold)} | Test samples: {len(X_test_fold)}")

        # Crea e addestra un nuovo modello KNN per questo specifico fold.
        knn_model = knn_model_class(X_train_fold, Y_train_fold, k_neighbors)

        # Esegue le predizioni sul set di test del fold corrente.
        y_pred = knn_model.test(X_test_fold)
        y_pred_proba = knn_model.test_proba(X_test_fold)

        # Calcola le metriche di performance per questo fold.
        fold_metrics = calculate_metrics(Y_test_fold, y_pred, y_pred_proba)

        # Aggiunge le metriche del fold alla lista complessiva.
        all_fold_metrics.append(fold_metrics)

        # Salva i dati grezzi per i plot specifici del fold
        all_fold_raw_data.append({
            'y_true': Y_test_fold,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })

    print("\nK-Fold Cross Validation completata.")

    # 3. RESTITUZIONE DEI RISULTATI
    # Ritorna un dizionario contenente solo la lista delle metriche per ogni fold.
    return {
        "all_fold_metrics": all_fold_metrics,
        "all_fold_raw_data": all_fold_raw_data
    }


def kfold_validation(X, Y, k, K_folds):
    """
    Esegue il workflow completo di validazione K-Fold.

    Args:
        X: Features (DataFrame o lista di liste)
        Y: Target (Series o lista)
        k: Numero di vicini per KNN
        K_folds: Numero di fold
    """
    # Assicura che i dati siano in formato lista
    X_data = X.values.tolist() if hasattr(X, 'values') else X
    Y_data = Y.values.tolist() if hasattr(Y, 'values') else Y

    results = evaluate_kfold(X_data, Y_data, KNN, k, K_folds)

    # Crea un prefisso unico per i file di output di questa esecuzione
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"kfold_k={k}_folds={K_folds}_{timestamp}"

    handler = KFoldResultsHandler(
        all_fold_metrics=results['all_fold_metrics'],
        all_fold_raw_data=results['all_fold_raw_data'],
        filename_prefix=prefix,
        y_true_all=results.get('y_true'),
        y_pred_all=results.get('y_pred'),
        y_pred_proba_all=results.get('y_pred_proba')
    )
    handler.save_results()