import unittest
import polars as pl
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from box import Box

import utils.predict_utils as ut

class TestPredictUtils(unittest.TestCase):

    def test_threshold_label_high(self):
        df = pl.DataFrame({'label': [1, 2, 3, 4, 5]})
        df_thresholded = ut.threshold_label(df, 'label', high=True)
        self.assertListEqual(df_thresholded.columns, ['label', 'label_raw'])
        self.assertEqual(df_thresholded['label'].dtype, pl.Boolean)

    def test_threshold_label_low(self):
        df = pl.DataFrame({'label': [0, 1, 0, 1, 0]})
        df_thresholded = ut.threshold_label(df, 'label', high=False)
        self.assertListEqual(df_thresholded.columns, ['label', 'label_raw'])
        self.assertEqual(df_thresholded['label'].dtype, pl.Boolean)

    @patch("builtins.open", new_callable=mock_open, read_data="tasks:\n - cost\nwindows:\n - two_year\n")
    def test_load_config(self, mock_file):
        config = ut.load_config()
        self.assertIsInstance(config, Box)
        self.assertIn('tasks', config)
        self.assertIn('windows', config)

    def test_build_subsets(self):
        df_baseline = pl.DataFrame({
            'feature_age_recode': [20, 80, 60],
            'feature_sex_1': [1, 0, 1],
            'feature_sex_2': [0, 1, 0],
            'feature_num_episodes': [1, 3, 6],
            'diagnosis_history': [['I10', 'E11'], ['N18'], ['F30']],
        })
        subsets = ut.build_subsets(df_baseline)
        self.assertIsInstance(subsets, dict)
        self.assertIn('over_75', subsets)
        self.assertIn('under_75', subsets)
        self.assertIn('female', subsets)
        self.assertIn('male', subsets)
        self.assertIn('num_episodes=1', subsets)
        self.assertIn('num_episodes=2-4', subsets)
        self.assertIn('num_episodes>=5', subsets)
        self.assertIn('cvd', subsets)
        self.assertIn('diabetes', subsets)
        self.assertIn('ckd', subsets)
        self.assertIn('mental_health', subsets)

    @patch("pandas.DataFrame.to_csv")
    def test_save_results(self, mock_to_csv):
        results_df_all = [pd.DataFrame({'auc': [0.8], 'window': 'one_year'})]
        config_dict = {'result_fout_format': 'results_{task}_{high}.csv'}
        config = Box(config_dict)
        ut.save_results(results_df_all, 'test_task', True, config)
        mock_to_csv.assert_called()

    def test_format_results(self):
        results = [{'subset': 'full_test', 'auc': 0.85, 'ap': 0.75, 'accuracy': 0.90, 'recall': 0.80, 'specificity': 0.95, 'brier': 0.10, 'dcal': 1.2}]
        results_df = ut.format_results(results)
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn('*', results_df['auc'][0])
        self.assertIn('*', results_df['ap'][0])
        self.assertIn('*', results_df['accuracy'][0])
        self.assertIn('*', results_df['recall'][0])
        self.assertIn('*', results_df['specificity'][0])
        self.assertIn('*', results_df['brier'][0])
        self.assertIn('*', results_df['dcal'][0])

    def test_verify_data_alignment_aligned(self):
        df_label = pl.DataFrame({'ppn_int': [1, 2, 3], 'label': [0, 1, 0]})
        embedding_set = {
            'embed1': pl.DataFrame({'ppn_int': [1, 2, 3], 'feature1': [0.1, 0.2, 0.3]}),
            'embed2': pl.DataFrame({'ppn_int': [1, 2, 3], 'feature2': [0.4, 0.5, 0.6]})
        }
        try:
            ut.verify_data_alignment(embedding_set, df_label)
        except AssertionError:
            self.fail("verify_data_alignment raised AssertionError unexpectedly for aligned data")

    def test_verify_data_alignment_size_mismatch(self):
        df_label = pl.DataFrame({'ppn_int': [1, 2, 3], 'label': [0, 1, 0]})
        embedding_set = {
            'embed1': pl.DataFrame({'ppn_int': [1, 2], 'feature1': [0.1, 0.2]})
        }
        with self.assertRaises(AssertionError):
            ut.verify_data_alignment(embedding_set, df_label)

    def test_verify_data_alignment_ppn_mismatch(self):
        df_label = pl.DataFrame({'ppn_int': [1, 2, 3], 'label': [0, 1, 0]})
        embedding_set = {
            'embed1': pl.DataFrame({'ppn_int': [1, 4, 3], 'feature1': [0.1, 0.2, 0.3]})
        }
        with self.assertRaises(AssertionError):
            ut.verify_data_alignment(embedding_set, df_label)

    def test_verify_data_alignment_unsorted(self):
        df_label = pl.DataFrame({'ppn_int': [1, 2, 3], 'label': [0, 1, 0]})
        embedding_set = {
            'embed1': pl.DataFrame({'ppn_int': [2, 1, 3], 'feature1': [0.1, 0.2, 0.3]})
        }
        with self.assertRaises(AssertionError):
            ut.verify_data_alignment(embedding_set, df_label)