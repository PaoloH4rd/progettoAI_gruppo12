"""
Test per il modulo ModelEvaluation/holdout_validation.py

Questi test verificano la correttezza dell'implementazione della validazione Holdout:
- Esecuzione corretta della pipeline di validazione
- Suddivisione stratificata del dataset
- Passaggio corretto dei dati al results handler
- Gestione di diverse percentuali di test
"""

import unittest
from unittest.mock import patch, MagicMock
from ModelEvaluation.holdout_validation import holdout_validation


class TestHoldoutValidation(unittest.TestCase):
    """Test per la funzione holdout_validation."""

    def setUp(self):
        """Prepara i dati di test comuni a tutti i test."""
        # Dataset di esempio con 10 campioni (5 per classe)
        self.X = [
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],  # Classe 0
            [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]  # Classe 1
        ]
        self.Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        # Metriche mock restituite da calculate_metrics
        self.mock_metrics = {
            'accuracy': 0.85,
            'error_rate': 0.15,
            'sensitivity': 0.90,
            'specificity': 0.80,
            'gmean': 0.85,
            'auc': 0.88
        }

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_executes_complete_pipeline(self, mock_random, mock_time, mock_knn_class,
                                         mock_calc_metrics, mock_handler_class):
        """Verifica che holdout_validation esegua correttamente la pipeline completa."""


        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)
        mock_time.strftime.return_value = "20260218_120000"

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1]
        mock_knn_instance.test_proba.return_value = [0.2, 0.8]
        mock_knn_class.return_value = mock_knn_instance

        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Act
        holdout_validation(self.X, self.Y, k=3, test_perc=0.2)

        # Assert
        mock_knn_class.assert_called_once()
        mock_knn_instance.test.assert_called_once()
        mock_knn_instance.test_proba.assert_called_once()
        mock_calc_metrics.assert_called_once()
        mock_handler_class.assert_called_once()

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_knn_receives_correct_k_parameter(self, mock_random, mock_time, mock_knn_class,
                                               mock_calc_metrics, mock_handler_class):
        """Verifica che KNN riceva il valore corretto di k."""

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)
        mock_time.strftime.return_value = "20260218_120000"

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1]
        mock_knn_instance.test_proba.return_value = [0.2, 0.8]
        mock_knn_class.return_value = mock_knn_instance

        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Act
        holdout_validation(self.X, self.Y, k=5, test_perc=0.2)

        # Assert
        call_args = mock_knn_class.call_args[0]
        self.assertEqual(call_args[2], 5)  # k=5

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_stratified_split_maintains_class_proportions(self, mock_random, mock_time,
                                                           mock_knn_class, mock_calc_metrics,
                                                           mock_handler_class):
        """Verifica che la suddivisione stratificata mantenga le proporzioni delle classi."""

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)
        mock_time.strftime.return_value = "20260218_120000"

        captured_train_data = {}

        def capture_knn_init(x_train, y_train, k):
            captured_train_data['Y_train'] = y_train
            mock_instance = MagicMock()
            mock_instance.test.return_value = [0, 1]
            mock_instance.test_proba.return_value = [0.3, 0.7]
            return mock_instance

        mock_knn_class.side_effect = capture_knn_init
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Act
        holdout_validation(self.X, self.Y, k=3, test_perc=0.2)

        # Assert
        y_train = captured_train_data['Y_train']
        self.assertIn(0, y_train, "Il training set deve contenere la classe 0")
        self.assertIn(1, y_train, "Il training set deve contenere la classe 1")
        self.assertEqual(len(y_train), 8)  # 80% di 10 campioni

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_handler_receives_correct_data(self, mock_random, mock_time, mock_knn_class,
                                            mock_calc_metrics, mock_handler_class):
        """Verifica che HoldoutResultsHandler riceva i dati corretti."""

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)

        expected_y_pred = [0, 1, 0, 1]
        expected_y_proba = [0.1, 0.9, 0.2, 0.8]

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = expected_y_pred
        mock_knn_instance.test_proba.return_value = expected_y_proba
        mock_knn_class.return_value = mock_knn_instance

        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_instance = MagicMock()
        mock_handler_class.return_value = mock_handler_instance

        # Act
        holdout_validation(self.X, self.Y, k=3, test_perc=0.2)

        # Assert
        handler_call_kwargs = mock_handler_class.call_args[1]
        self.assertEqual(handler_call_kwargs['metrics'], self.mock_metrics)
        self.assertEqual(handler_call_kwargs['y_pred'], expected_y_pred)
        self.assertEqual(handler_call_kwargs['y_pred_proba'], expected_y_proba)

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_handles_dataframe_input(self, mock_random, mock_time, mock_knn_class,
                                      mock_calc_metrics, mock_handler_class):
        """Verifica che la funzione gestisca correttamente input DataFrame/Series."""

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)

        mock_X = MagicMock()
        mock_X.values.tolist.return_value = self.X

        mock_Y = MagicMock()
        mock_Y.values.tolist.return_value = self.Y

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1]
        mock_knn_instance.test_proba.return_value = [0.3, 0.7]
        mock_knn_class.return_value = mock_knn_instance

        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Act
        holdout_validation(mock_X, mock_Y, k=3, test_perc=0.2)

        # Assert
        mock_X.values.tolist.assert_called_once()
        mock_Y.values.tolist.assert_called_once()

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_different_test_percentages_produce_correct_sizes(self, mock_random, mock_time,
                                                               mock_knn_class, mock_calc_metrics,
                                                               mock_handler_class):
        """Verifica che diverse percentuali di test producano le dimensioni corrette."""

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)

        captured_train_size = []

        def capture_knn_init(x_train, y_train, k):
            captured_train_size.append(len(x_train))
            mock_instance = MagicMock()
            mock_instance.test.return_value = [0]
            mock_instance.test_proba.return_value = [0.5]
            return mock_instance

        mock_knn_class.side_effect = capture_knn_init
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Act
        holdout_validation(self.X, self.Y, k=3, test_perc=0.3)

        # Assert: con 30% test -> 70% train = 8 campioni (per stratificazione)
        self.assertEqual(captured_train_size[0], 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
