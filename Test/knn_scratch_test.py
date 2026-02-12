import unittest
from ModelDevelopment.knn_scratch import KNN


class TestKNN(unittest.TestCase):

    def setUp(self):
        """Inizializza i dati di test per ogni test case."""
        self.x_train = [[1, 2], [2, 3], [3, 4], [6, 7]]
        self.y_train = [0, 0, 1, 1]
        self.k = 3
        self.knn = KNN(self.x_train, self.y_train, self.k)

    def test_init(self):
        """Testa l'inizializzazione del modello KNN."""
        self.assertEqual(self.knn.x_train, self.x_train)
        self.assertEqual(self.knn.y_train, self.y_train)
        self.assertEqual(self.knn.k, self.k)

    def test_euclidean_distance(self):
        """Testa il calcolo della distanza euclidea."""
        x_test = [[2, 2], [5, 6]]
        distances = self.knn.euclidean_distance(x_test)

        self.assertEqual(len(distances), 2)
        self.assertEqual(len(distances[0]), 4)
        self.assertAlmostEqual(distances[0][0], 1.0, places=2)

    def test_test_method(self):
        """Testa il metodo di predizione."""
        x_test = [[2, 2], [5, 6]]
