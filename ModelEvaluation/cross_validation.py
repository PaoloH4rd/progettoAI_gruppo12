import random
from collections import defaultdict
from ModelEvaluation.metrics import calculate_metrics


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
        # Un defaultdict(list) si comporta come un dizionario normale, ma se si tenta di acceder a una chiave che non esiste, la crea automaticamente con una lista vuota come valore.
        # Questo è utile per evitare di dover controllare ogni volta se una chiave (la classe) è già presente nel dizionario prima di aggiungere un elemento.
        indices_by_class = defaultdict(list)

        # 2. Raggruppa gli indici dei dati per classe
        # Itera su tutta la lista delle etichette (Y), tenendo traccia sia dell'indice (i) che del valore dell'etichetta (label) per ogni campione.

        for i, label in enumerate(Y):
            # Usa l'etichetta (es. 2 per 'Benigno', 4 per 'Maligno') come chiave del dizionario
            # e aggiunge l'indice (la posizione) del campione a quella lista.
            # Alla fine di questo ciclo, avremo un dizionario simile a questo:
            # {
            #   2: [0, 2, 5, ...],  <-- Tutti gli indici dei campioni benigni
            #   4: [1, 3, 4, ...]   <-- Tutti gli indici dei campioni maligni
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


        # UTILIZZO l'asterisco per passare a zip le chiavi e i valori di train_combinedd invece chel'oggetto in se
        X_train, Y_train = zip(*train_combined) if train_combined else ([], [])

        # Esegue la stessa operazione di mescolamento per il test set.
        test_combined = list(zip(X_test, Y_test))
        random.shuffle(test_combined)
        X_test, Y_test = zip(*test_combined) if test_combined else ([], [])

        # 6. Restituzione dei dati
        # Converte le tuple restituite da zip in liste e le restituisce.
        return list(X_train), list(X_test), list(Y_train), list(Y_test)

