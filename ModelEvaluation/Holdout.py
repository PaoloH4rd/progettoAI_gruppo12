
import random

class Holdout:
    def __init__(self, test_size, random_state):
        """
        Inizializza la classe Holdout con la dimensione del test set e il seed random.

        Args:
            test_size: percentuale per il test set (default 0.2)
            random_state: seed per riproducibilità (opzionale)
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_train_test(self, X, Y):
        """
        Divide il dataset in training e test set.

        Args:
            X: features (lista o array)
            Y: labels (lista o array)

        Returns:
            X_train, X_test, Y_train, Y_test
        """

        # Imposta il seed se fornito
        if self.random_state is not None:
            random.seed(self.random_state)

        # Crea indici e mescolali
        indices = list(range(len(X)))
        random.shuffle(indices)

        # Calcola l'indice di split
        split_idx = int(len(indices) * (1 - self.test_size))

        # Dividi gli indici
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Crea i set
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        Y_train = [Y[i] for i in train_indices]
        Y_test = [Y[i] for i in test_indices]

        return X_train, X_test, Y_train, Y_test
