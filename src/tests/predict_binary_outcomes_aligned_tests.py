import numpy as np
import polars as pl
import sklearn

# Import the functions you want to test from predict_binary_outcomes_aligned.py
from downstream_prediction.predict_binary_outcomes_aligned import (
    eval_cv_model,
    get_embeddings,
)
from evaluation.metrics import specificity_score
from utils.predict_utils import threshold_label

# If predict_binary_outcomes_aligned.py is in a parent directory, you might need to add it to the Python path
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from downstream_prediction.predict_binary_outcomes_aligned import ...

import unittest
from unittest.mock import patch, MagicMock

class TestPredictBinaryOutcomesAligned(unittest.TestCase):

    def test_specificity_score(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        self.assertAlmostEqual(specificity_score(y_true, y_pred), 0.5)

    @patch('sklearn.model_selection.BaseCrossValidator')
    @patch('sklearn.pipeline.Pipeline')
    def test_eval_cv_model(self, mock_pipeline, mock_cv):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        mock_cv.split.return_value = [(np.array([0, 1]), np.array([2, 3]))]
        mock_pipeline.return_value.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        
        models = [('TestModel', MagicMock(), {})]
        result = eval_cv_model(models, X, y, mock_cv, 'one_year', 'test_feature_set')
        
        self.assertIn('metrics', result)
        #self.assertIn('predictions', result)
        self.assertEqual(len(result['metrics']), 1)
        #self.assertIn('TestModel_y_pred', result['predictions'])
        #self.assertIn('TestModel_y_true', result['predictions'])

    def test_threshold_label(self):
        df = pl.DataFrame({
            'cost': [100, 200, 300, 400, 500],
            'other_task': [1, 2, 3, 4, 5]
        })
        
        result = threshold_label(df, 'cost', high=True)
        self.assertTrue(all(result['cost'] == [False, False, False, True, True]))
        
        result = threshold_label(df, 'other_task', high=False)
        self.assertTrue(all(result['other_task'] == [True, True, True, True, True]))

    @patch('polars.scan_parquet')
    def test_get_embeddings(self, mock_scan_parquet):
        mock_df = pl.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        mock_scan_parquet.return_value.collect.return_value = mock_df
        
        result = get_embeddings('mock.parquet')
        self.assertTrue(result.frame_equal(mock_df))

if __name__ == '__main__':
    unittest.main()
