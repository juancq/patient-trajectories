import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from box import Box

from evaluation.metrics import (calculate_metrics, d_calibration,
                                specificity_score)
import utils.predict_utils as ut
import prediction.predict_binary_outcomes_aligned as main
from prediction.predict_binary_outcomes_aligned import (add_diagnosis_codes, eval_cv_model, eval_model_train_test,
                  evaluate_subsets, get_embeddings, get_models,
                  perform_cross_validation, process_embeddings)


class TestMain(unittest.TestCase):

    def setUp(self):
        self.mock_config = Box({
            'label_path_nt': 'label_path',
            'label_path_linux': 'label_path',
            'test_size': None,
            'seed': 42,
            'stratified': True,
            'n_splits': 2,
            'tasks': ['task1'],
            'windows': ['two_year'],
            'result_fout_format': 'results_{task}_{high}.csv',
            'run_label': 'test_run',
        })
        self.mock_args = MagicMock(no_cv=False, high=True)
        self.patch_load_config = patch('prediction.predict_binary_outcomes_aligned.ut.load_config', return_value=self.mock_config)
        self.patch_argparse = patch('prediction.predict_binary_outcomes_aligned.argparse.ArgumentParser')
        self.patch_logger_info = patch('prediction.predict_binary_outcomes_aligned.logger.info')
        self.patch_joblib_dump = patch('prediction.predict_binary_outcomes_aligned.joblib.dump')
        self.patch_pd_to_csv = patch('pandas.DataFrame.to_csv')
        self.patch_calculate_metrics = patch('prediction.predict_binary_outcomes_aligned.calculate_metrics', return_value={'auc': 0.8})
        self.patch_calibration_display = patch('prediction.predict_binary_outcomes_aligned.CalibrationDisplay.from_predictions')
        self.patch_plt_savefig = patch('prediction.predict_binary_outcomes_aligned.plt.savefig')
        self.patch_conditional_set_memory_limit = patch('prediction.predict_binary_outcomes_aligned.conditional_set_memory_limit', lambda x: lambda f: f)

        self.mock_config_context = patch('prediction.predict_binary_outcomes_aligned.sklearn.config_context')
        self.mock_sklearnex_config_context = patch('prediction.predict_binary_outcomes_aligned.sklearnex_config_context')

        self.mock_sklearnex_config_context.start()
        self.mock_config_context.start()
        self.patch_load_config.start()
        self.patch_argparse.start()
        self.patch_logger_info.start()
        self.patch_joblib_dump.start()
        self.patch_pd_to_csv.start()
        self.patch_calculate_metrics.start()
        self.patch_calibration_display.start()
        self.patch_plt_savefig.start()
        self.patch_conditional_set_memory_limit.start()


    def tearDown(self):
        self.mock_sklearnex_config_context.stop()
        self.mock_config_context.stop()
        self.patch_load_config.stop()
        self.patch_argparse.stop()
        self.patch_logger_info.stop()
        self.patch_joblib_dump.stop()
        self.patch_pd_to_csv.stop()
        self.patch_calculate_metrics.stop()
        self.patch_calibration_display.stop()
        self.patch_plt_savefig.stop()
        self.patch_conditional_set_memory_limit.stop()

    @patch('prediction.predict_binary_outcomes_aligned.StandardScaler')
    @patch('prediction.predict_binary_outcomes_aligned.Pipeline')
    @patch('prediction.predict_binary_outcomes_aligned.joblib.dump')
    @patch('pandas.DataFrame.to_csv')
    @patch('prediction.predict_binary_outcomes_aligned.calculate_metrics', return_value={'auc': 0.8})
    def test_eval_cv_model(self, mock_calculate_metrics, mock_to_csv, mock_joblib_dump, MockPipeline, MockStandardScaler):
        models = [('LR', LogisticRegression, {})]
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        fold_indices = [(np.array([0, 1]), np.array([2, 3]))]
        window = 'two_year'
        feature_set = 'test_features'
        subset_data = {}

        mock_scaler = MockStandardScaler.return_value
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])
        MockPipeline.return_value = mock_pipeline_instance

        results = eval_cv_model(models, X, y, fold_indices, window, feature_set, subset_data=subset_data)
        self.assertIsInstance(results, dict)
        self.assertIn('metrics', results)
        self.assertEqual(len(results['metrics']), 1)
        self.assertEqual(results['metrics'][0]['model'], 'LR')
        mock_calculate_metrics.assert_called()
        mock_joblib_dump.assert_called()
        mock_to_csv.assert_called()


    def test_add_diagnosis_codes(self):
        df = pl.DataFrame({
            'ppn_int': [1, 1, 2, 2, 3],
            'diagnosis_history': [['A01.0', 'B02'], ['C03.2'], ['A01.0', 'D04.3'], ['E05.4', 'F06.5'], ['G07.6']],
            'height': [170, 170, 180, 180, 190]
        })
        df_with_codes = add_diagnosis_codes(df)
        print(df_with_codes)
        self.assertIn('feature_diagnosis_A01.0', df_with_codes.columns)
        self.assertIn('feature_diagnosis_B02', df_with_codes.columns)
        self.assertIn('feature_diagnosis_C03.2', df_with_codes.columns)
        self.assertIn('feature_diagnosis_D04.3', df_with_codes.columns)
        self.assertIn('feature_diagnosis_E05.4', df_with_codes.columns)
        self.assertIn('feature_diagnosis_F06.5', df_with_codes.columns)
        self.assertIn('feature_diagnosis_G07.6', df_with_codes.columns) 

    def test_get_embeddings_file_not_found(self):
        with self.assertRaises(ValueError):
            get_embeddings('non_existent_file.parquet')

    def test_get_models(self):
        models = get_models()
        self.assertIsInstance(models, list)
        self.assertEqual(len(models), 4)
        model_names = [m[0] for m in models]
        self.assertIn('LR', model_names)
        self.assertIn('lgbm', model_names)
        self.assertIn('ElasticNet', model_names)
        self.assertIn('Dummy', model_names)

    @patch('prediction.predict_binary_outcomes_aligned.eval_cv_model')
    def test_perform_cross_validation(self, mock_eval_cv_model):
        embedding_set = {
            'baseline_diagnosis': pl.DataFrame({
                'ppn_int': [1, 2],
                'feature1': [0.1, 0.2],
                'task1': [0, 1]
            })
        }
        task = 'task1'
        fold_indices = [(np.array([0]), np.array([1]))]
        models = get_models()
        config = self.mock_config
        window = 'two_year'
        subset_data = {}

        results = perform_cross_validation(embedding_set, task, fold_indices, models, config, window, subset_data)
        self.assertIsInstance(results, list)
        mock_eval_cv_model.assert_called()