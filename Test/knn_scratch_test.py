"""
Test per il modulo ModelDevelopment/knn_scratch.py

Questi test verificano la correttezza dell'implementazione del KNN:
- Inizializzazione corretta del modello
- Calcolo della distanza euclidea
- Predizione delle classi
- Calcolo delle probabilità
"""

import unittest
from ModelDevelopment.knn_scratch import KNN


class TestKNN(unittest.TestCase):
    """Test per la classe KNN."""

    def setUp(self):
        """Prepara i dati di test comuni a tutti i test."""
        # Arrange: Dataset di esempio con 4 campioni
        self.x_train = [[1, 2], [2, 3], [3, 4], [6, 7]]
        self.y_train = [0, 0, 1, 1]
        self.k = 3
        self.knn = KNN(self.x_train, self.y_train, self.k)

    def test_init_stores_training_data(self):
        """Verifica che il costruttore memorizzi correttamente i dati di training."""
        # Assert
        self.assertEqual(self.knn.x_train, self.x_train)
        self.assertEqual(self.knn.y_train, self.y_train)
        self.assertEqual(self.knn.k, self.k)

    def test_euclidean_distance_returns_correct_shape(self):
        """Verifica che euclidean_distance restituisca una matrice della dimensione corretta."""
        # Arrange
        x_test = [[2, 2], [5, 6]]
        distances = self.knn.euclidean_distance(x_test)

        # Assert: 2 campioni di test x 4 campioni di training
        self.assertEqual(len(distances), 2)
        self.assertEqual(len(distances[0]), 4)

    def test_euclidean_distance_calculates_correctly(self):
        """Verifica che la distanza euclidea sia calcolata correttamente."""
        # Arrange: punto [2,2] rispetto a [1,2] -> distanza = sqrt((2-1)^2 + (2-2)^2) = 1.0
        x_test = [[2, 2]]

        # Act
        distances = self.knn.euclidean_distance(x_test)

        # Assert
        self.assertAlmostEqual(distances[0][0], 1.0, places=2)

    def test_euclidean_distance_zero_for_same_point(self):
        """Verifica che la distanza di un punto da se stesso sia 0."""
        # Arrange
        x_test = [[1, 2]]  # Stesso punto del primo campione di training

        # Act
        distances = self.knn.euclidean_distance(x_test)

        # Assert
        self.assertAlmostEqual(distances[0][0], 0.0, places=5)

    def test_test_method_returns_correct_number_of_predictions(self):
        """Verifica che test() restituisca una predizione per ogni campione."""
        # Arrange
        x_test = [[2, 2], [5, 6]]

        # Act
        predictions = self.knn.test(x_test)

        # Assert
        self.assertEqual(len(predictions), 2)

    def test_test_method_returns_valid_classes(self):
        """Verifica che le predizioni siano classi valide (0 o 1)."""
        # Arrange
        x_test = [[2, 2], [5, 6]]

        # Act
        predictions = self.knn.test(x_test)

        # Assert
        for pred in predictions:
            self.assertIn(pred, [0, 1])

    def test_test_proba_returns_probabilities(self):
        """Verifica che test_proba restituisca probabilità nel range [0, 1]."""
        # Arrange
        x_test = [[2, 2], [5, 6]]

        # Act
        probabilities = self.knn.test_proba(x_test)

        # Assert
        self.assertEqual(len(probabilities), 2)
        for prob in probabilities:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_prediction_near_class_0_samples(self):
        """Verifica che un punto vicino ai campioni di classe 0 venga classificato come 0."""
        # Arrange: [1.5, 2.5] è vicino a [1,2] e [2,3] che sono classe 0
        x_test = [[1.5, 2.5]]

        # Act
        predictions = self.knn.test(x_test)

        # Assert
        self.assertEqual(predictions[0], 0)

    def test_prediction_near_class_1_samples(self):
        """Verifica che un punto vicino ai campioni di classe 1 venga classificato come 1."""
        # Arrange: [5.5, 6.5] è vicino a [6,7] che è classe 1
        x_test = [[5.5, 6.5]]

        # Act
        predictions = self.knn.test(x_test)

        # Assert
        self.assertEqual(predictions[0], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
