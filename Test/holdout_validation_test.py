import unittest
from unittest.mock import patch, MagicMock


class TestHoldoutValidation(unittest.TestCase):
    """
    Test per la funzione holdout_validation usando mock per isolare le dipendenze.

    Mock utilizzati:
    - KNN: il modello di classificazione
    - calculate_metrics: calcolo delle metriche di valutazione
    - HoldoutResultsHandler: salvataggio dei risultati
    - random: per rendere i test deterministici
    - time.strftime: per timestamp prevedibili
    """

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
            'precision': 0.80,
            'recall': 0.90,
            'f1_score': 0.85,
            'auc_roc': 0.88
        }

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_holdout_validation_basic(self, mock_random, mock_time, mock_knn_class,
                                       mock_calc_metrics, mock_handler_class):
        """
        Test base: verifica che holdout_validation esegua correttamente
        la pipeline di validazione con i parametri forniti.
        """
        from ModelEvaluation.holdout_validation import holdout_validation

        # Configura i mock
        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)  # Non mescola
        mock_time.strftime.return_value = "20260210_120000"

        # Mock del modello KNN
        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = [0, 1]  # Predizioni
        mock_knn_instance.test_proba.return_value = [0.2, 0.8]  # Probabilità
        mock_knn_class.return_value = mock_knn_instance

        # Mock delle metriche
        mock_calc_metrics.return_value = self.mock_metrics

        # Mock del handler
        mock_handler_instance = MagicMock()
        mock_handler_class.return_value = mock_handler_instance

        # Esegui la funzione
        holdout_validation(self.X, self.Y, k=3, test_perc=0.2)

        # Verifica che KNN sia stato creato con i parametri corretti
        mock_knn_class.assert_called_once()
        call_args = mock_knn_class.call_args
        self.assertEqual(call_args[0][2], 3)  # k=3

        # Verifica che test e test_proba siano stati chiamati
        mock_knn_instance.test.assert_called_once()
        mock_knn_instance.test_proba.assert_called_once()

        # Verifica che calculate_metrics sia stato chiamato
        mock_calc_metrics.assert_called_once()

        # Verifica che il handler sia stato creato e save_results chiamato
        mock_handler_class.assert_called_once()
        mock_handler_instance.save_results.assert_called_once()

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_stratified_split(self, mock_random, mock_time, mock_knn_class,
                               mock_calc_metrics, mock_handler_class):
        """
        Test: verifica che la suddivisione sia stratificata,
        cioè che la proporzione delle classi sia mantenuta in train e test.
        """
        from ModelEvaluation.holdout_validation import holdout_validation

        # Non mescolare per rendere il test deterministico
        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)
        mock_time.strftime.return_value = "20260210_120000"

        # Cattura i dati passati a KNN
        captured_train_data = {}

        def capture_knn_init(x_train, y_train, k):
            captured_train_data['X_train'] = x_train
            captured_train_data['Y_train'] = y_train
            captured_train_data['k'] = k
            mock_instance = MagicMock()
            mock_instance.test.return_value = [0, 1]
            mock_instance.test_proba.return_value = [0.3, 0.7]
            return mock_instance

        mock_knn_class.side_effect = capture_knn_init
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Esegui con 20% test (2 campioni per classe nel test)
        holdout_validation(self.X, self.Y, k=5, test_perc=0.2)

        # Verifica che il training set abbia campioni di entrambe le classi
        y_train = captured_train_data['Y_train']
        self.assertIn(0, y_train, "Il training set deve contenere la classe 0")
        self.assertIn(1, y_train, "Il training set deve contenere la classe 1")

        # Con 10 campioni e 20% test, dovremmo avere ~8 nel training
        self.assertEqual(len(y_train), 8)

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_handler_receives_correct_data(self, mock_random, mock_time, mock_knn_class,
                                            mock_calc_metrics, mock_handler_class):
        """
        Test: verifica che HoldoutResultsHandler riceva i dati corretti
        (metriche, predizioni, probabilità).
        """
        from ModelEvaluation.holdout_validation import holdout_validation

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)

        # Predizioni e probabilità specifiche
        expected_y_pred = [0, 1, 0, 1]
        expected_y_proba = [0.1, 0.9, 0.2, 0.8]

        mock_knn_instance = MagicMock()
        mock_knn_instance.test.return_value = expected_y_pred
        mock_knn_instance.test_proba.return_value = expected_y_proba
        mock_knn_class.return_value = mock_knn_instance

        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_instance = MagicMock()
        mock_handler_class.return_value = mock_handler_instance

        holdout_validation(self.X, self.Y, k=3, test_perc=0.2)

        # Verifica i parametri passati al handler
        handler_call_kwargs = mock_handler_class.call_args[1]

        self.assertEqual(handler_call_kwargs['metrics'], self.mock_metrics)
        self.assertEqual(handler_call_kwargs['y_pred'], expected_y_pred)
        self.assertEqual(handler_call_kwargs['y_pred_proba'], expected_y_proba)
        self.assertIn('holdout_k=3', handler_call_kwargs['filename_prefix'])

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_with_dataframe_input(self, mock_random, mock_time, mock_knn_class,
                                   mock_calc_metrics, mock_handler_class):
        """
        Test: verifica che la funzione gestisca correttamente
        input in formato DataFrame/Series (con attributo .values).
        """
        from ModelEvaluation.holdout_validation import holdout_validation

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)
        # Simula DataFrame e Series con attributo .values
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

        # Non deve sollevare eccezioni
        holdout_validation(mock_X, mock_Y, k=3, test_perc=0.2)

        # Verifica che .values.tolist() sia stato chiamato
        mock_X.values.tolist.assert_called_once()
        mock_Y.values.tolist.assert_called_once()

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_different_test_percentages(self, mock_random, mock_time, mock_knn_class,
                                         mock_calc_metrics, mock_handler_class):
        """
        Test: verifica che diverse percentuali di test producano
        le corrette dimensioni di train/test set.
        """
        from ModelEvaluation.holdout_validation import holdout_validation

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)

        captured_sizes = []

        def capture_knn_init(x_train, y_train, k):
            captured_sizes.append(len(x_train))
            mock_instance = MagicMock()
            mock_instance.test.return_value = [0]
            mock_instance.test_proba.return_value = [0.5]
            return mock_instance

        mock_knn_class.side_effect = capture_knn_init
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Test con 30% test set (7 train, 3 test per 10 campioni)
        holdout_validation(self.X, self.Y, k=3, test_perc=0.3)

        # Con 10 campioni e 30% test, 1 campione per classe va al test (int(5*0.3)=1)
        # Quindi 4 per classe nel training = 8 totali
        self.assertEqual(captured_sizes[0], 8)

    @patch('ModelEvaluation.holdout_validation.HoldoutResultsHandler')
    @patch('ModelEvaluation.holdout_validation.calculate_metrics')
    @patch('ModelEvaluation.holdout_validation.KNN')
    @patch('ModelEvaluation.holdout_validation.time')
    @patch('ModelEvaluation.holdout_validation.random')
    def test_empty_test_set(self, mock_random, mock_time, mock_knn_class,
                            mock_calc_metrics, mock_handler_class):
        """
        Test: verifica il comportamento con test_perc=0 (tutto in training).
        """
        from ModelEvaluation.holdout_validation import holdout_validation

        mock_random.seed = MagicMock()
        mock_random.shuffle = MagicMock(side_effect=lambda x: None)

        captured_data = {}

        def capture_knn_init(x_train, y_train, k):
            captured_data['train_size'] = len(x_train)
            mock_instance = MagicMock()
            mock_instance.test.return_value = []
            mock_instance.test_proba.return_value = []
            return mock_instance

        mock_knn_class.side_effect = capture_knn_init
        mock_calc_metrics.return_value = self.mock_metrics
        mock_handler_class.return_value = MagicMock()

        # Con 0% test, tutti i dati sono nel training
        holdout_validation(self.X, self.Y, k=3, test_perc=0.0)

        self.assertEqual(captured_data['train_size'], 10)


if __name__ == '__main__':
    unittest.main()
