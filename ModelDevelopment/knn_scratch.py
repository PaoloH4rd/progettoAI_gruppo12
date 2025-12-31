class KNN:

    def __init__(self, x_train, y_train, k):
        """
        Costruttore che inizializza le caratteristiche dei dati di addestramento, le etichette e il numero di vicini.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = k


    def euclidean_distance(self, x_test):
        """

        Calcola la distanza euclidea tra i dati di addestramento e di test.

        Args:
        x_test (list): Lista di caratteristiche dei dati di test.

        Returns:
        list: Lista di distanze euclidee tra ogni campione di test e tutti i campioni di addestramento.
        """
        test_dists = []
        for i in range(len(x_test)):
            dists = [
                (sum([(float(x_test[i][j]) - float(self.x_train[m][j])) ** 2 for j in range(len(x_test[i]))])) ** 0.5
                for m in range(len(self.x_train))]
            test_dists.append(dists)
        return test_dists


    def test(self, x_test):
        """
        Testa il modello sui dati di test e fa delle predizioni.

        Args:
        x_test (list): Lista di caratteristiche dei dati di test.

        Returns:
        list: Lista delle tabelle predette per i dati di test.
        """
        y_test_pred = []
        test_dists = self.euclidean_distance(x_test)  # Trova la Distanza Euclidea tra i dati di test e di training
        for dists in test_dists:
            k_smallest = sorted(range(len(dists)), key=lambda i: dists[i])[:self.k]
            labels = [self.y_train[i] for i in k_smallest]
            label_counts = {}
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            y_test_pred.append(max(label_counts, key=label_counts.get))

        return y_test_pred

    def test_proba(self, x_test):
        """
        Calcola le probabilità predette per i dati di test.
        Ritorna la probabilità della classe positiva (la classe con valore più alto).

        Args:
        x_test (list): Lista di caratteristiche dei dati di test.

        Returns:
        list: Lista delle probabilità per la classe positiva per ogni campione di test.
        """
        y_test_proba = []
        test_dists = self.euclidean_distance(x_test)  #  Questo metodo calcola
        # la distanza euclidea tra i campioni di test forniti in x_test e tutti
        # i campioni di addestramento memorizzati in self.x_train.

        # Identifica le classi uniche e assume che la classe con valore più alto sia la "positiva".
        unique_classes = list(set(self.y_train))
        positive_class = max(unique_classes)

        for dists in test_dists:
            # Trova gli indici dei k vicini più prossimi (distanza minore).
            k_smallest = sorted(range(len(dists)), key=lambda i: dists[i])[:self.k]
            
            # Recupera le etichette di questi k vicini dal training set.
            labels = [self.y_train[i] for i in k_smallest]
            
            # Conta le occorrenze di ogni etichetta tra i vicini.
            label_counts = {}
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            
            # Calcola la probabilità della classe positiva come frazione dei vicini positivi su k.
            proba_positive = label_counts.get(positive_class, 0) / self.k
            y_test_proba.append(proba_positive)

        return y_test_proba
