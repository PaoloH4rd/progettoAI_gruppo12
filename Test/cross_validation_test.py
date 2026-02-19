"""
Test per il modulo ModelEvaluation/cross_validation.py

Questi test verificano la correttezza dell'implementazione della K-Fold Cross Validation:
- Creazione del numero corretto di fold
- Dimensioni corrette di train e test set
- Restituzione delle metriche per ogni fold
"""

import unittest
from unittest.mock import patch, MagicMock
from ModelEvaluation.cross_validation import k_fold_split, evaluate_kfold


class TestKFoldSplit(unittest.TestCase):
    """Test per la funzione k_fold_split."""

    def setUp(self):
        """Prepara i dati di test comuni a tutti i test."""
        self.X = [[i] for i in range(100)]
        self.Y = [i % 2 for i in range(100)]

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_creates_correct_number_of_folds(self, mock_shuffle):
        """Verifica che venga creato il numero corretto di fold."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 5

        # Act
        folds = k_fold_split(self.X, self.Y, k_folds)

        # Assert
        self.assertEqual(len(folds), k_folds)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_each_fold_has_train_and_test(self, mock_shuffle):
        """Verifica che ogni fold abbia 4 elementi: X_train, Y_train, X_test, Y_test."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 5

        # Act
        folds = k_fold_split(self.X, self.Y, k_folds)

        # Assert
        for fold in folds:
            self.assertEqual(len(fold), 4, "Ogni fold deve avere 4 elementi")
            X_train, Y_train, X_test, Y_test = fold
            self.assertIsInstance(X_train, list)
            self.assertIsInstance(Y_train, list)
            self.assertIsInstance(X_test, list)
            self.assertIsInstance(Y_test, list)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_correct_train_test_sizes(self, mock_shuffle):
        """Verifica che le dimensioni di train e test siano corrette (80/20 per 5 fold)."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 5
        expected_test_size = len(self.X) // k_folds  # 20

        # Act
        folds = k_fold_split(self.X, self.Y, k_folds)

        # Assert
        for fold in folds:
            X_train, Y_train, X_test, Y_test = fold
            self.assertEqual(len(X_test), expected_test_size)
            self.assertEqual(len(X_train), len(self.X) - expected_test_size)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_no_data_leakage_between_train_and_test(self, mock_shuffle):
        """Verifica che non ci sia sovrapposizione tra train e test."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 5

        # Act
        folds = k_fold_split(self.X, self.Y, k_folds)

        # Assert
        for fold in folds:
            X_train, Y_train, X_test, Y_test = fold
            train_set = set(map(tuple, X_train))
            test_set = set(map(tuple, X_test))
            intersection = train_set & test_set
            self.assertEqual(len(intersection), 0,
                           "Train e test non devono avere elementi in comune")


class TestEvaluateKFold(unittest.TestCase):
    """Test per la funzione evaluate_kfold."""

    def setUp(self):
        """Prepara i dati di test comuni a tutti i test."""
        self.X = [[i] for i in range(100)]
        self.Y = [i % 2 for i in range(100)]

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_returns_metrics_for_each_fold(self, mock_shuffle):
        """Verifica che vengano restituite metriche per ogni fold."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 5
        k_neighbors = 3

        mock_knn_class = MagicMock()
        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1] * 10
        mock_knn_instance.test_proba.return_value = [0.3, 0.7] * 10
        mock_knn_class.return_value = mock_knn_instance

        # Act
        results = evaluate_kfold(self.X, self.Y, mock_knn_class, k_neighbors, k_folds)

        # Assert
        self.assertIn("all_fold_metrics", results)
        self.assertIn("all_fold_raw_data", results)
        self.assertEqual(len(results["all_fold_metrics"]), k_folds)
        self.assertEqual(len(results["all_fold_raw_data"]), k_folds)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_knn_created_for_each_fold(self, mock_shuffle):
        """Verifica che KNN venga creato una volta per ogni fold."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 5
        k_neighbors = 3

        mock_knn_class = MagicMock()
        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1] * 10
        mock_knn_instance.test_proba.return_value = [0.3, 0.7] * 10
        mock_knn_class.return_value = mock_knn_instance

        # Act
        evaluate_kfold(self.X, self.Y, mock_knn_class, k_neighbors, k_folds)

        # Assert
        self.assertEqual(mock_knn_class.call_count, k_folds)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_knn_receives_correct_k(self, mock_shuffle):
        """Verifica che KNN riceva il valore corretto di k."""
        # Arrange
        mock_shuffle.side_effect = lambda x: x
        k_folds = 3
        k_neighbors = 7

        mock_knn_class = MagicMock()
        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1] * 10
        mock_knn_instance.test_proba.return_value = [0.3, 0.7] * 10
        mock_knn_class.return_value = mock_knn_instance

        # Act
        evaluate_kfold(self.X, self.Y, mock_knn_class, k_neighbors, k_folds)

        # Assert
        for call in mock_knn_class.call_args_list:
            self.assertEqual(call[0][2], k_neighbors)


if __name__ == '__main__':
    unittest.main(verbosity=2)
