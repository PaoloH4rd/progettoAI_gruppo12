"""
Test per il modulo ModelEvaluation/metrics.py

Questi test verificano la correttezza delle formule matematiche per:
- Accuracy: corretti / totale
- Error Rate: 1 - accuracy
- Sensitivity (Recall/TPR): TP / (TP + FN)
- Specificity (TNR): TN / (TN + FP)
- Geometric Mean: sqrt(sensitivity * specificity)
- AUC: Area Under ROC Curve
"""

import unittest
import math
import random
from ModelEvaluation.metrics import build_confusion_matrix, calculate_metrics


class TestBuildConfusionMatrixExtended(unittest.TestCase):
    """Test estesi per la funzione build_confusion_matrix."""

    def test_mixed_predictions_all_values(self):
        """Verifica che predizioni miste producano i valori corretti della matrice."""
        # Arrange: 2 TP, 2 TN, 1 FP, 1 FN
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]

        # Act
        tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)

        # Assert
        self.assertEqual(tp, 2)
        self.assertEqual(tn, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

    def test_all_false_positives(self):
        """Verifica il caso in cui tutti i negativi sono classificati come positivi."""
        # Arrange
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]

        # Act
        tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)

        # Assert
        self.assertEqual(tp, 0)
        self.assertEqual(tn, 0)
        self.assertEqual(fp, 4)
        self.assertEqual(fn, 0)

    def test_all_false_negatives(self):
        """Verifica il caso in cui tutti i positivi sono classificati come negativi."""
        # Arrange
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]

        # Act
        tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)

        # Assert
        self.assertEqual(tp, 0)
        self.assertEqual(tn, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 4)

    def test_empty_lists(self):
        """Verifica il comportamento con liste vuote."""
        # Arrange
        y_true = []
        y_pred = []

        # Act
        tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)

        # Assert
        self.assertEqual(tp, 0)
        self.assertEqual(tn, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)


class TestCalculateMetrics(unittest.TestCase):
    """Test per la funzione calculate_metrics."""

    def test_perfect_classifier(self):
        """Verifica che un classificatore perfetto abbia accuracy=1 e error_rate=0."""
        # Arrange
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        y_proba = [0.1, 0.2, 0.9, 0.95, 0.15, 0.85]

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['error_rate'], 0.0)
        self.assertEqual(metrics['sensitivity'], 1.0)
        self.assertEqual(metrics['specificity'], 1.0)

    def test_accuracy_formula(self):
        """Verifica che accuracy = corretti / totale."""
        # Arrange: 4 corretti su 6 totali
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        y_proba = [0.8, 0.4, 0.2, 0.7, 0.9, 0.3]
        expected_accuracy = 4 / 6

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy, places=4)

    def test_error_rate_formula(self):
        """Verifica che error_rate = 1 - accuracy."""
        # Arrange
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        y_proba = [0.8, 0.4, 0.2, 0.7, 0.9, 0.3]
        expected_error_rate = 2 / 6

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertAlmostEqual(metrics['error_rate'], expected_error_rate, places=4)

    def test_sensitivity_formula(self):
        """Verifica che sensitivity = TP / (TP + FN)."""
        # Arrange: 2 TP, 1 FN -> sensitivity = 2/3
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        y_proba = [0.8, 0.4, 0.2, 0.7, 0.9, 0.3]
        expected_sensitivity = 2 / 3

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertAlmostEqual(metrics['sensitivity'], expected_sensitivity, places=4)

    def test_specificity_formula(self):
        """Verifica che specificity = TN / (TN + FP)."""
        # Arrange: 2 TN, 1 FP -> specificity = 2/3
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        y_proba = [0.8, 0.4, 0.2, 0.7, 0.9, 0.3]
        expected_specificity = 2 / 3

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertAlmostEqual(metrics['specificity'], expected_specificity, places=4)

    def test_geometric_mean_formula(self):
        """Verifica che gmean = sqrt(sensitivity * specificity)."""
        # Arrange
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        y_proba = [0.8, 0.4, 0.2, 0.7, 0.9, 0.3]
        expected_gmean = math.sqrt((2/3) * (2/3))

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertAlmostEqual(metrics['gmean'], expected_gmean, places=4)

    def test_all_wrong_predictions(self):
        """Verifica che un classificatore che sbaglia tutto abbia accuracy=0."""
        # Arrange
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        y_proba = [0.9, 0.8, 0.1, 0.2]

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertEqual(metrics['accuracy'], 0.0)
        self.assertEqual(metrics['error_rate'], 1.0)

    def test_no_positive_predictions(self):
        """Verifica il comportamento quando non ci sono predizioni positive."""
        # Arrange: tutti predetti come 0
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_proba = [0.3, 0.4, 0.1, 0.2]

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertEqual(metrics['sensitivity'], 0.0)  # 0 TP, 2 FN
        self.assertEqual(metrics['specificity'], 1.0)  # 2 TN, 0 FP

    def test_auc_is_calculated(self):
        """Verifica che AUC venga calcolato correttamente."""
        # Arrange
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        y_proba = [0.1, 0.2, 0.9, 0.95, 0.15, 0.85]

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertIn('auc', metrics)
        self.assertIsNotNone(metrics['auc'])
        self.assertGreater(metrics['auc'], 0.9)


class TestMetricsEdgeCases(unittest.TestCase):
    """Test per casi limite delle metriche."""

    def test_single_sample(self):
        """Verifica il comportamento con un solo campione."""
        # Arrange
        y_true = [1]
        y_pred = [1]
        y_proba = [0.9]

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertEqual(metrics['accuracy'], 1.0)

    def test_large_dataset(self):
        """Verifica che le metriche siano calcolate correttamente su dataset grandi."""

        random.seed(42)
        n = 1000
        y_true = [random.randint(0, 1) for _ in range(n)]
        y_pred = [random.randint(0, 1) for _ in range(n)]
        y_proba = [random.random() for _ in range(n)]

        # Act
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Assert
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_metrics_without_proba(self):
        """Verifica che le metriche base vengano calcolate anche senza probabilità."""
        # Arrange
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]

        # Act
        metrics = calculate_metrics(y_true, y_pred)

        # Assert
        self.assertIn('accuracy', metrics)
        self.assertEqual(metrics['accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

