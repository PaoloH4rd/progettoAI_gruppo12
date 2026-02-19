"""
Test per il modulo ModelEvaluation/stratified_shuffle_split_validation.py

Questi test verificano la correttezza dell'implementazione dello Stratified Shuffle Split nel caso di validazione multipla:
- Numero corretto di esperimenti generati
- Passaggio corretto dei parametri a KNN
- Verifica che il results handler venga creato e utilizzato
"""

import unittest
from unittest.mock import patch, MagicMock
from ModelEvaluation.stratified_shuffle_split_validation import stratified_shuffle_split_validation


class TestStratifiedShuffleSplitValidation(unittest.TestCase):
    """Test per la funzione stratified_shuffle_split_validation."""

    def setUp(self):
        """Prepara i dati di test comuni a tutti i test."""
        self.X = [[i, i+1] for i in range(100)]
        self.Y = [0] * 50 + [1] * 50
        self.mock_metrics = {
            'accuracy': 0.8,
            'error_rate': 0.2,
            'sensitivity': 0.85,
            'specificity': 0.75,
            'gmean': 0.80,
            'auc': 0.82
        }

    @patch('ModelEvaluation.stratified_shuffle_split_validation.StratifiedShuffleSplitResultsHandler')
    @patch('ModelEvaluation.stratified_shuffle_split_validation.calculate_metrics')
    @patch('ModelEvaluation.stratified_shuffle_split_validation.KNN')
    @patch('builtins.print')
    def test_runs_correct_number_of_experiments(self, mock_print, mock_knn_class, mock_calc_metrics, mock_handler_class):
        """Verifica che vengano eseguiti il numero corretto di esperimenti."""

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1] * 10
        mock_knn_instance.test_proba.return_value = [0.3, 0.7] * 10
        mock_knn_class.return_value = mock_knn_instance
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        n_experiments = 5

        # Act
        stratified_shuffle_split_validation(self.X, self.Y, k=3, n_experiments=n_experiments)

        # Assert
        self.assertEqual(mock_knn_class.call_count, n_experiments)

    @patch('ModelEvaluation.stratified_shuffle_split_validation.StratifiedShuffleSplitResultsHandler')
    @patch('ModelEvaluation.stratified_shuffle_split_validation.calculate_metrics')
    @patch('ModelEvaluation.stratified_shuffle_split_validation.KNN')
    @patch('builtins.print')
    def test_knn_receives_correct_k(self, mock_print, mock_knn_class, mock_calc_metrics, mock_handler_class):
        """Verifica che KNN riceva il valore corretto di k."""

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1] * 10
        mock_knn_instance.test_proba.return_value = [0.3, 0.7] * 10
        mock_knn_class.return_value = mock_knn_instance
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        k_value = 7

        # Act
        stratified_shuffle_split_validation(self.X, self.Y, k=k_value, n_experiments=2)

        # Assert
        for call in mock_knn_class.call_args_list:
            self.assertEqual(call[0][2], k_value)

    @patch('ModelEvaluation.stratified_shuffle_split_validation.StratifiedShuffleSplitResultsHandler')
    @patch('ModelEvaluation.stratified_shuffle_split_validation.calculate_metrics')
    @patch('ModelEvaluation.stratified_shuffle_split_validation.KNN')
    def test_results_handler_is_called(self, mock_knn_class, mock_calc_metrics, mock_handler_class):
        """Verifica che il results handler venga creato e utilizzato."""

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1] * 10
        mock_knn_instance.test_proba.return_value = [0.3, 0.7] * 10
        mock_knn_class.return_value = mock_knn_instance
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_instance = MagicMock()
        mock_handler_class.return_value = mock_handler_instance

        # Act
        stratified_shuffle_split_validation(self.X, self.Y, k=3, n_experiments=2)

        # Assert
        mock_handler_class.assert_called_once()
        mock_handler_instance.save_results.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
