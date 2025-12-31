import time
import numpy as np

# Assicurati che questi import funzionino nel tuo progetto
from ModelDevelopment.knn_scratch import KNN
from ModelEvaluation.metrics import calculate_metrics
from ModelEvaluation.results_handler import StratifiedShuffleSplitResultsHandler


def binary_stratified_shuffle_split(Y, n_experiments=1, test_size=0.2, random_seed=50):
    """
    Generatore procedurale per Stratified Shuffle Split su 2 Classi (0 e 1).
    Restituisce gli INDICI di train e test.
    """
    Y = np.array(Y)
    rng = np.random.default_rng(random_seed)
    n_samples = len(Y)
    #creo un array di indici da 0 a n_samples-1
    indices = np.arange(n_samples)

    # SEPARAZIONE: Creiamo due liste di indici fisse all'inizio
    # Assumiamo che le classi siano 0 e 1 (o comunque binarie e ordinate)
    classes = [0, 1]

    indices_0 = indices[Y == classes[0]]  # Solitamente classe 0 o Negativa (benigno)
    indices_1 = indices[Y == classes[1]]  # Solitamente classe 1 o Positiva (maligno)

    # Calcoliamo subito quanti prenderne per il test da ciascun gruppo
    n_test_0 = int(len(indices_0) * test_size)
    n_test_1 = int(len(indices_1) * test_size)

    # Ciclo per il numero di split richiesti
    for _ in range(n_experiments):
        # 2. MESCOLAMENTO INDIPENDENTE (Shuffle)
        # Mescoliamo le copie degli indici per non alterare gli originali
        current_idx_0 = rng.permutation(indices_0)
        current_idx_1 = rng.permutation(indices_1)

        # Split
        # Preleviamo la quota per il test dalla classe 0
        test_0 = current_idx_0[:n_test_0]
        train_0 = current_idx_0[n_test_0:]

        # Preleviamo la quota per il test dalla classe 1
        test_1 = current_idx_1[:n_test_1]
        train_1 = current_idx_1[n_test_1:]

        # Uniamo i pezzi: Test con Test, Train con Train
        final_test = np.concatenate([test_0, test_1])
        final_train = np.concatenate([train_0, train_1])

        # MESCOLAMENTO FINALE
        # Mischiamo il risultato finale per non avere ordine di classe
        rng.shuffle(final_test)
        rng.shuffle(final_train)

        # Restituiamo gli indici di train e test
        # yield restituisce un generatore ad ogni iterazione
        yield final_train, final_test


def stratified_shuffle_split_validation(X, Y, k, n_experiments):
    """
    Esegue la validazione utilizzando Stratified Shuffle Split.
    """
    # Assicuriamoci che siano numpy array per l'indicizzazione avanzata
    X = np.array(X)
    Y = np.array(Y)

    print(f"\nAvvio Stratified Shuffle Split con {n_experiments} esperimenti...")

    # Inizializziamo il generatore
    splitter = binary_stratified_shuffle_split(Y, n_experiments=n_experiments, test_size=0.2)

    all_experiment_metrics = []
    all_experiment_raw_data = []

    # Iteriamo sul generatore
    # Nota: enumerate parte da 1 solo per estetica nel print
    for i, (train_idx, test_idx) in enumerate(splitter, 1):

        # SLICING: Convertiamo gli indici in dati reali
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        print(f"  - Iterazione {i}/{n_experiments}")
        print(f"    Training samples: {len(X_train)} | Test samples: {len(X_test)}")

        # Controllo rapido proporzione classi 0 / 1  nel test set
        prop_test = np.sum(Y_test == 1) / len(Y_test)
        print(f"    Proporzione Classe 1  (maligni) nel Test: {prop_test:.2%}")

        # Addestramento e test + probabilità
        knn_model = KNN(X_train, Y_train, k)
        y_pred = knn_model.test(X_test)
        y_pred_proba = knn_model.test_proba(X_test)

        # Metriche
        metrics = calculate_metrics(Y_test, y_pred, y_pred_proba)
        all_experiment_metrics.append(metrics)

        # Dati grezzi per i grafici
        all_experiment_raw_data.append({
            'y_true': Y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })

    print("\nValutazione completata.")

    # Salvataggio Risultati
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"shuffle_split_k={k}_n={n_experiments}_{timestamp}"

    handler = StratifiedShuffleSplitResultsHandler(
        all_experiment_metrics=all_experiment_metrics,
        all_experiment_raw_data=all_experiment_raw_data,
        filename_prefix=prefix
    )
    handler.save_results()
    print(f"Risultati salvati con prefisso: {prefix}")
