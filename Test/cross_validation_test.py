import unittest
from unittest.mock import patch, Mock, MagicMock
from ModelEvaluation.cross_validation import k_fold_split, evaluate_kfold, kfold_validation


class TestKFoldSplit(unittest.TestCase):
    """Test per la funzione k_fold_split"""

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_k_fold_split_creates_correct_number_of_folds(self, mock_shuffle):
        """Verifica che venga creato il numero corretto di fold"""
        # Mock shuffle per avere un comportamento deterministico
        mock_shuffle.side_effect = lambda x: x

        X = [[i] for i in range(100)]
        Y = [i % 2 for i in range(100)]
        k_folds = 5

        folds = k_fold_split(X, Y, k_folds)

        self.assertEqual(len(folds), k_folds)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_k_fold_split_correct_sizes(self, mock_shuffle):
        """Verifica che le dimensioni dei set di train e test siano corrette"""
        mock_shuffle.side_effect = lambda x: x

        X = [[i] for i in range(100)]
        Y = [i % 2 for i in range(100)]
        k_folds = 5

        folds = k_fold_split(X, Y, k_folds)

    @patch('ModelEvaluation.cross_validation.random.shuffle')
    def test_evaluation_kfold_returns_metrics(self, mock_shuffle):
        """Verifica che evaluate_kfold ritorni un dizionario con le metriche per ogni fold"""
        mock_shuffle.side_effect = lambda x: x

        X = [[i] for i in range(100)]
        Y = [i % 2 for i in range(100)]
        k_folds = 5
        k_neighbors = 3

        # Mock del modello KNN
        mock_knn_model_class = Mock()
        mock_knn_instance = Mock()
        mock_knn_instance.test.return_value = [0, 1, 0, 1, 0]
        mock_knn_instance.test_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.95, 0.05]]
        mock_knn_model_class.return_value = mock_knn_instance

        results = evaluate_kfold(X, Y, mock_knn_model_class, k_neighbors, k_folds)

        self.assertIn("all_fold_metrics", results)
        self.assertIn("all_fold_raw_data", results)
        self.assertEqual(len(results["all_fold_metrics"]), k_folds)
        self.assertEqual(len(results["all_fold_raw_data"]), k_folds)