import unittest
import numpy as np

from evaluation.metrics import (calculate_metrics, d_calibration,
                                specificity_score)
class TestMetrics(unittest.TestCase):

    def test_specificity_score(self):
        y_true = np.array([0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 1, 0])
        specificity = specificity_score(y_true, y_pred)
        self.assertAlmostEqual(specificity, 0.75)

    def test_specificity_score_perfect(self):
        y_true = np.array([0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0])
        specificity = specificity_score(y_true, y_pred)
        self.assertAlmostEqual(specificity, 1.0)

    def test_specificity_score_worst(self):
        y_true = np.array([0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 1])
        specificity = specificity_score(y_true, y_pred)
        self.assertAlmostEqual(specificity, 0.0)

    def test_d_calibration(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
        d_cal, p_value = d_calibration(y_true, y_pred)
        self.assertIsInstance(d_cal, float)
        self.assertIsInstance(p_value, float)

    def test_d_calibration_perfect_calibration(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        d_cal, p_value = d_calibration(y_true, y_pred)
        self.assertIsInstance(d_cal, float)
        self.assertIsInstance(p_value, float)

    def test_calculate_metrics(self):
        y_test = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
        metrics = calculate_metrics(y_test, y_pred)
        self.assertIsInstance(metrics, dict)
        self.assertIn('auc', metrics)
        self.assertIn('ap', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('specificity', metrics)
        self.assertIn('brier', metrics)
        self.assertIn('dcal', metrics)
        self.assertIn('dcal_pvalue', metrics)

    def test_calculate_metrics_single_class(self):
        y_test = np.array([0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
        metrics = calculate_metrics(y_test, y_pred)
        self.assertIsNone(metrics)