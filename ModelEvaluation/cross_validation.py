import random
from collections import defaultdict
from .metrics import calculate_metrics


# noinspection PyPep8Naming
class KFoldCrossValidation:
    def __init__(self, k=5):
        self.k = k

    def holdout_split(self, X, Y, test_size=0.2, random_state=42):
        """
        Args:
            X (list): Lista di feature
            Y (list): Lista di label
            test_size (float): Proporzione del dataset da allocare al test set
            random_state (int): Seed per la riproducibilità
        """
        if random_state is not None:
            random.seed(random_state)

        # 1. Inizializza un dizionario speciale (defaultdict)
        # Un defaultdict(list) si comporta come un dizionario normale, ma se si tenta di accedere a una chiave che non esiste, la crea automaticamente con una lista vuota come valore.
        # Questo è utile per evitare di dover controllare ogni volta se una chiave (la classe) è già presente nel dizionario prima di aggiungere un elemento.
        indices_by_class = defaultdict(list)

        # 2. Raggruppa gli indici dei dati per classe
        # Itera su tutta la lista delle etichette (Y), tenendo traccia sia dell'indice (i) che del valore dell'etichetta (label) per ogni campione.
        for i, label in enumerate(Y):
            # Usa l'etichetta (es. 2 per 'Benigno', 4 per 'Maligno') come chiave del dizionario
            # e aggiunge l'indice (la posizione) del campione a quella lista.
            # Alla fine di questo ciclo, avremo un dizionario simile a questo:
            # {
            #    2: [0, 2, 5, ...],  <-- Tutti gli indici dei campioni benigni
            #    4: [1, 3, 4, ...]   <-- Tutti gli indici dei campioni maligni
            # }
            indices_by_class[label].append(i)

        X_train, Y_train, X_test, Y_test = [], [], [], []

        # 3. Suddivisione stratificata per ogni classe
        # Itera su ogni classe (es. 2 e 4) e sulla lista di indici corrispondenti.
        for label, indices in indices_by_class.items():
            # Mescola gli indici di quella classe per garantire una selezione casuale.
            random.shuffle(indices)
            # Calcola il numero di campioni da destinare al test set per questa classe.
            n_test = int(len(indices) * test_size)
            # Seleziona gli indici per il test set.
            test_indices = indices[:n_test]
            # I restanti indici sono per il training set.
            train_indices = indices[n_test:]

            # 4. Popolamento dei set di training e test
            for i in train_indices:
                X_train.append(X[i])
                Y_train.append(Y[i])
            for i in test_indices:
                X_test.append(X[i])
                Y_test.append(Y[i])

        # 5. Mescolamento finale dei dati
        # Poiché i dati sono stati aggiunti classe per classe, ora sono ordinati (es. tutti i benigni, poi tutti i maligni).
        # È buona pratica mescolarli per evitare che l'ordine influenzi l'addestramento del modello.

        # Combina le feature e le etichette di training in coppie.
        train_combined = list(zip(X_train, Y_train))
        # Mescola le coppie.
        random.shuffle(train_combined)
        # Separa nuovamente le feature e le etichette, ora mescolate.
        # Il controllo 'if train_combined' evita errori se il set di training è vuoto.

        # UTILIZZO l'asterisco per passare a zip le chiavi e i valori di train_combined invece che l'oggetto in se
        X_train, Y_train = zip(*train_combined) if train_combined else ([], [])

        # Esegue la stessa operazione di mescolamento per il test set.
        test_combined = list(zip(X_test, Y_test))
        random.shuffle(test_combined)
        X_test, Y_test = zip(*test_combined) if test_combined else ([], [])

        # 6. Restituzione dei dati
        # Converte le tuple restituite da zip in liste e le restituisce.
        return list(X_train), list(X_test), list(Y_train), list(Y_test)

    def split_data(self, X, Y):
        """
        Suddivide i dati in k fold per la K-Fold Cross Validation standard.
        Ogni fold viene utilizzato una volta come set di test, mentre i restanti k-1 fold
        vengono usati come set di addestramento.

        Args:
            X (list): Lista di feature.
            Y (list): Lista di label.

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
        fold_size = len(X) // self.k
        # Lista per contenere i dati di tutti i fold.
        folds = []

        # 3. Creazione di ogni fold
        # Itera k volte, una per ogni fold da creare.
        for i in range(self.k):
            # 4. Definizione degli indici di test per il fold corrente
            # Calcola l'indice di inizio del blocco di test.
            test_start = i * fold_size
            # Calcola l'indice di fine. L'ultimo fold prende tutti gli indici rimanenti
            # per gestire i casi in cui la dimensione del dataset non è perfettamente divisibile per k.
            test_end = test_start + fold_size if i < self.k - 1 else len(X)
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

    def evaluate(self, X, Y, knn_model_class, k_neighbors):
        """
        Esegue una validazione K-Fold  sull'intero dataset.
        1. Suddivide l'INTERO dataset in K parti (fold).
        2. Per ogni iterazione (fold), usa 1 parte come Test Set e le restanti K-1 come Training Set.
        3. Calcola le metriche per ognuno dei K esperimenti e le restituisce.
        """
        # 1. PREPARAZIONE PER LA K-FOLD CROSS VALIDATION
        # Suddivide l'intero dataset (X, Y) in 'k' fold.
        # Questo assicura che ogni singolo esempio del dataset venga usato esattamente una volta per il test.
        folds = self.split_data(X, Y)
        all_fold_metrics = []

        print(f"\n{'=' * 60}")
        print(f"INIZIO K-FOLD CROSS VALIDATION (k={self.k})")
        print(f"Totale campioni nel dataset: {len(X)}")
        print(f"{'=' * 60}\n")

        # 2. ESECUZIONE DELLA K-FOLD CROSS VALIDATION
        # Itera su ogni fold. Ad ogni iterazione, un fold diverso viene usato come test set
        # e i restanti k-1 fold vengono usati come training set.
        for fold_num, (X_train_fold, Y_train_fold, X_test_fold, Y_test_fold) in enumerate(folds, 1):
            print(f"  - Esperimento {fold_num}/{self.k}")
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

        print("\nK-Fold Cross Validation completata.")

        # 3. RESTITUZIONE DEI RISULTATI
        # Ritorna un dizionario contenente solo la lista delle metriche per ogni fold.
        return {
            "all_fold_metrics": all_fold_metrics
        }