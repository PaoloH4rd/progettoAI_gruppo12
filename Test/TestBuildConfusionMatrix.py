import unittest
from ModelEvaluation.metrics import build_confusion_matrix


class TestBuildConfusionMatrix(unittest.TestCase):
    """Test per build_confusion_matrix"""

    def test_all_true_positives(self):
        """Tutti i campioni sono positivi correttamente classificati"""

        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)
        self.assertEqual(tp, 4)
        self.assertEqual(tn, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_all_true_negatives(self):
        """Tutti i campioni sono negativi correttamente classificati"""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)
        self.assertEqual(tp, 0)
        self.assertEqual(tn, 4)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)