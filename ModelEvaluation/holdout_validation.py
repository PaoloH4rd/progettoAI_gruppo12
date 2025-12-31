import time
import random
from collections import defaultdict
from ModelDevelopment.knn_scratch import KNN
from ModelEvaluation.metrics import calculate_metrics
from ModelEvaluation.results_handler import HoldoutResultsHandler


def holdout_validation(X, Y, k, test_perc):
    """
    Esegue il workflow completo di validazione Holdout.

    Args:
        X: Features (DataFrame o lista di liste)
        Y: Target (Series o lista)
        k: Numero di vicini per KNN
        test_perc: Percentuale del test set (0.0 - 1.0)

    Returns:
        None
    """

    # Assicura che i dati siano in formato lista (se passati come DataFrame/Series da pandas)
    X_data = X.values.tolist() if hasattr(X, 'values') else X
    Y_data = Y.values.tolist() if hasattr(Y, 'values') else Y

    random.seed(42) # Seed fisso per riproducibilità

    # Inizializza un dizionario speciale (defaultdict)
    # Un defaultdict(list) si comporta come un dizionario normale, ma se si tenta di accedere a una chiave che non esiste, la crea automaticamente con una lista vuota come valore.
    # Questo è utile per evitare di dover controllare ogni volta se una chiave (la classe) è già presente nel dizionario prima di aggiungere un elemento.
    indices_by_class = defaultdict(list)

    # Raggruppa gli indici dei dati per classe
    # Itera su tutta la lista delle etichette (Y), tenendo traccia sia dell'indice (i) che del valore dell'etichetta (label) per ogni campione.
    for i, label in enumerate(Y_data):
        # Usa l'etichetta (es. 2 per 'Benigno', 4 per 'Maligno') come chiave del dizionario
        # e aggiunge l'indice (la posizione) del campione a quella lista.
        # Alla fine di questo ciclo, avremo un dizionario simile a questo:
        # {
        #    2: [0, 2, 5, ...],  <-- Tutti gli indici dei campioni benigni
        #    4: [1, 3, 4, ...]   <-- Tutti gli indici dei campioni maligni
        # }
        indices_by_class[label].append(i)

    X_train, Y_train, X_test, Y_test = [], [], [], []

    # Suddivisione stratificata per ogni classe
    # Itera su ogni classe (es. 2 e 4) e sulla lista di indici corrispondenti.
    for label, indices in indices_by_class.items():
        # Mescola gli indici di quella classe per garantire una selezione casuale.
        random.shuffle(indices)
        # Calcola il numero di campioni da destinare al test set per questa classe.
        n_test = int(len(indices) * test_perc)
        # Seleziona gli indici per il test set.
        test_indices = indices[:n_test]
        # I restanti indici sono per il training set.
        train_indices = indices[n_test:]

        # Popolamento dei set di training e test
        for i in train_indices:
            X_train.append(X_data[i])
            Y_train.append(Y_data[i])
        for i in test_indices:
            X_test.append(X_data[i])
            Y_test.append(Y_data[i])

    # Mescolamento finale dei dati (Training)
    # Poiché i dati sono stati aggiunti classe per classe, ora sono ordinati (es. tutti i benigni, poi tutti i maligni).
    # È buona pratica mescolarli per evitare che l'ordine influenzi l'addestramento del modello.

    # Combina le feature e le etichette di training in coppie.
    train_combined = list(zip(X_train, Y_train))
    # Mescola le coppie.
    random.shuffle(train_combined)
    # Separa nuovamente le feature e le etichette, ora mescolate.
    # Il controllo 'if train_combined' evita errori se il set di training è vuoto.
    if train_combined:
        # UTILIZZO l'asterisco per passare a zip le chiavi e i valori di train_combined invece che l'oggetto in se
        X_train_tup, Y_train_tup = zip(*train_combined)
        X_train = list(X_train_tup)
        Y_train = list(Y_train_tup)
    else:
        X_train, Y_train = [], []

    # Mescolamento finale dei dati (Test)
    # Esegue la stessa operazione di mescolamento per il test set.
    test_combined = list(zip(X_test, Y_test))
    random.shuffle(test_combined)
    if test_combined:
        X_test_tup, Y_test_tup = zip(*test_combined)
        X_test = list(X_test_tup)
        Y_test = list(Y_test_tup)
    else:
        X_test, Y_test = [], []

    print(f"\n--- Divisione Holdout ({int((1-test_perc)*100)}/{int(test_perc*100)}) ---")
    print(f"Dimensioni Training Set: {len(X_train)} campioni")
    print(f"Dimensioni Test Set: {len(X_test)} campioni")
    print("------------------------------------")

    # Addestramento
    print("\nAddestramento del modello KNN...")
    knn_model = KNN(X_train, Y_train, k)
    print("Addestramento completato.")

    # Valutazione
    print("\nValutazione del modello sul Test Set...")
    y_pred = knn_model.test(X_test)
    y_pred_proba = knn_model.test_proba(X_test)
    print("Valutazione completata.")

    # Calcolo metriche
    metrics = calculate_metrics(Y_test, y_pred, y_pred_proba)

    # Salvataggio Risultati
    # Crea un prefisso unico per i file di output di questa esecuzione
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"holdout_k={k}_{timestamp}"

    handler = HoldoutResultsHandler(
        metrics=metrics,
        y_true=Y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        filename_prefix=prefix
    )
    handler.save_results()