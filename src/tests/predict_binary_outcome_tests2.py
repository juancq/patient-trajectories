import unittest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock, mock_open
import pickle

# Add the parent directory to the path so we can import the target module
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predict_utils import threshold_label, load_config, build_subsets, verify_data_alignment
from evaluation.metrics import calculate_metrics, d_calibration, specificity_score
import utils.predict_utils as ut

# Import the main script - assuming this is in a tests/ directory
import prediction.predict_binary_outcomes_aligned as pred

class TestPredictBinaryOutcomes(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 2, 100)
        
        # Create sample fold indices
        self.fold_indices = [(np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9]))]
        
        # Create sample subset data
        self.subset_data = {
            'over_75': np.array([True, False, True, False, True] * 20),
            'female': np.array([False, True, False, True, False] * 20)
        }
        
        # Sample data for testing embeddings
        self.sample_embeddings = pl.DataFrame({
            'ppn_int': list(range(1, 101)),
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100)
        })
        
        # Sample data for testing baseline features
        self.sample_baseline = pl.DataFrame({
            'ppn_int': list(range(1, 101)),
            'feature_age_recode': np.random.randint(50, 90, 100),
            'feature_sex_1': np.random.randint(0, 2, 100),
            'feature_sex_2': np.random.randint(0, 2, 100),
            'feature_num_episodes': np.random.randint(1, 10, 100),
            'diagnosis_history': [[f"A{i}", f"B{i}", f"I15.{i%2}", f"E11.{i%3}"] for i in range(100)]
        })
        
        # Sample label data
        self.sample_label = pl.DataFrame({
            'ppn_int': list(range(1, 101)),
            'label': np.random.randint(0, 2, 100)
        })

    def test_evaluate_subsets(self):
        # Test the evaluate_subsets function
        y_test = self.y[5:10]  # Matches the test indices from fold_indices
        y_pred = np.random.rand(5)  # Predictions for the test set
        test_index = np.array([5, 6, 7, 8, 9])
        
        scores = {'subset1': {m: [] for m in ['auc', 'ap', 'accuracy', 'recall', 'specificity', 'brier', 'dcal', 'dcal_pvalue']}}
        
        # Create a patch for calculate_metrics
        with patch('prediction.predict_binary_outcomes_aligned.calculate_metrics') as mock_metrics:
            mock_metrics.return_value = {'auc': 0.8, 'ap': 0.7, 'accuracy': 0.75, 
                                          'recall': 0.6, 'specificity': 0.85, 
                                          'brier': 0.2, 'dcal': 5.0, 'dcal_pvalue': 1}
            
            pred.evaluate_subsets(
                {'subset1': self.subset_data['over_75']}, 
                y_test, 
                y_pred, 
                test_index, 
                scores
            )
            
            # Check if calculate_metrics was called with the right parameters
            mock_metrics.assert_called_once()
            # Check if scores were properly updated
            self.assertEqual(scores['subset1']['auc'], [0.8])
            self.assertEqual(scores['subset1']['ap'], [0.7])

    def test_eval_cv_model(self):
        # Test the eval_cv_model function
        models = [('TestModel', MagicMock(), {})]
        feature_set = 'test_features'
        label = 'test_label'
        window = 'two_year'
        
        # Mock pipeline and predict_proba
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9], [0.4, 0.6]])
        
        with patch('prediction.predict_binary_outcomes_aligned.Pipeline', return_value=mock_pipeline), \
             patch('prediction.predict_binary_outcomes_aligned.calculate_metrics') as mock_metrics, \
             patch('prediction.predict_binary_outcomes_aligned.joblib.dump') as mock_dump, \
             patch('prediction.predict_binary_outcomes_aligned.Path.mkdir'), \
             patch('pandas.DataFrame.to_csv'):
                
            mock_metrics.return_value = {'auc': 0.8, 'ap': 0.7, 'accuracy': 0.75, 
                                         'recall': 0.6, 'specificity': 0.85, 
                                         'brier': 0.2, 'dcal': 5.0, 'dcal_pvalue': 1}
                
            result = pred.eval_cv_model(
                models, 
                self.X, 
                self.y, 
                self.fold_indices, 
                window,
                feature_set, 
                label, 
                self.subset_data
            )
            
            # Check if the function returns the expected structure
            self.assertIn('metrics', result)
            self.assertIsInstance(result['metrics'], list)
            
            # Check that metrics were calculated
            mock_metrics.assert_called()
            
            # Check that model was saved
            mock_dump.assert_called()

    def test_add_diagnosis_codes(self):
        # Test the add_diagnosis_codes function
        with patch('polars.col'), \
             patch('polars.lit'), \
             patch('polars.LazyFrame.collect'), \
             patch('polars.LazyFrame.explode'), \
             patch('polars.LazyFrame.select'), \
             patch('polars.LazyFrame.filter'), \
             patch('polars.LazyFrame.group_by'), \
             patch('polars.LazyFrame.join'), \
             patch('polars.DataFrame.with_columns'), \
             patch('polars.DataFrame.select'):
                 
            # Just verify the function doesn't raise exceptions
            result = pred.add_diagnosis_codes(self.sample_baseline)
            # This test is more about making sure the function doesn't crash
            # rather than checking specific output values

    def test_get_embeddings(self):
        # Test the get_embeddings function
        with patch('os.path.isfile', return_value=True), \
             patch('polars.scan_parquet') as mock_scan, \
             patch('polars.LazyFrame.with_columns') as mock_with_columns, \
             patch('polars.LazyFrame.select') as mock_select:
                
            mock_scan.return_value = MagicMock()
            mock_with_columns.return_value = MagicMock()
            mock_select.return_value = MagicMock()
            
            # Configure the mocks to chain properly
            mock_scan.return_value.with_columns = mock_with_columns
            mock_with_columns.return_value.select = mock_select
            mock_select.return_value.collect.return_value.item.side_effect = [100, 1000]
            
            result = pred.get_embeddings("test_file.parquet")
            
            # Check that the function was called with the right parameters
            mock_scan.assert_called_with("test_file.parquet")
            
        # Test with non-existent file
        with patch('os.path.isfile', return_value=False), \
             self.assertRaises(ValueError):
            pred.get_embeddings("nonexistent_file.parquet")

    def test_get_models(self):
        # Test the get_models function
        models = pred.get_models()
        
        # Check if the function returns the expected number of models
        self.assertEqual(len(models), 4)
        
        # Check if the returned models have the expected structure
        for model_tuple in models:
            self.assertEqual(len(model_tuple), 3)
            name, model_factory, kwargs = model_tuple
            self.assertIsInstance(name, str)
            self.assertTrue(callable(model_factory) if name != 'Dummy' else True)
            self.assertIsInstance(kwargs, dict)

    def test_process_embeddings(self):
        # Test the process_embeddings function
        embed_path = {'test_embed': 'test_path_{window}.parquet'}
        window = 'two_year'
        label_ids = pl.DataFrame({'ppn_int': list(range(1, 101))}).lazy()
        
        with patch('prediction.predict_binary_outcomes_aligned.get_embeddings') as mock_get_embeddings, \
             patch('polars.LazyFrame.join') as mock_join, \
             patch('polars.LazyFrame.sort') as mock_sort, \
             patch('polars.LazyFrame.collect') as mock_collect, \
             patch('polars.DataFrame.select') as mock_select, \
             patch('pathlib.Path.__str__', return_value='test_path_two_year.parquet'):
                
            # Configure the mocks
            mock_get_embeddings.return_value = self.sample_embeddings.lazy()
            mock_join.return_value = self.sample_embeddings.lazy()
            mock_sort.return_value = self.sample_embeddings.lazy()
            mock_collect.return_value = self.sample_embeddings
            mock_select.return_value.item.return_value = 100
            
            result = pred.process_embeddings(embed_path, window, label_ids)
            
            # Check if the function returns a dictionary with the expected structure
            self.assertIsInstance(result, dict)
            self.assertIn('test_embed', result)
            
            # Check if get_embeddings was called with the right parameters
            mock_get_embeddings.assert_called_with('test_path_two_year.parquet')

    def test_perform_cross_validation(self):
        # Test the perform_cross_validation function
        embedding_set = {
            'baseline': self.sample_baseline.with_columns(pl.lit(0).alias('label')),
            'test_embed': self.sample_embeddings.with_columns(pl.lit(0).alias('label'))
        }
        task = 'label'
        models = [('TestModel', MagicMock(), {})]
        config = MagicMock()
        window = 'two_year'
        
        with patch('prediction.predict_binary_outcomes_aligned.eval_cv_model') as mock_eval_cv_model:
            mock_eval_cv_model.return_value = {'metrics': [{'subset': 'full_test', 'task': 'label', 'feature_set': 'baseline', 'model': 'TestModel', 'auc': 0.8}]}
            
            result = pred.perform_cross_validation(
                embedding_set, 
                task, 
                self.fold_indices, 
                models, 
                config, 
                window,
                self.subset_data
            )
            
            # Check if the function returns a list
            self.assertIsInstance(result, list)
            
            # Check if eval_cv_model was called the right number of times
            self.assertEqual(mock_eval_cv_model.call_count, 2)  # Once for each embedding

    def test_main_function(self):
        # Test the main function - this is mostly to ensure it doesn't crash
        # Rather than testing exact behavior
        with patch('argparse.ArgumentParser.parse_args') as mock_args, \
             patch('utils.predict_utils.load_config') as mock_load_config, \
             patch('prediction.predict_binary_outcomes_aligned.get_models') as mock_get_models, \
             patch('polars.scan_parquet') as mock_scan_parquet, \
             patch('prediction.predict_binary_outcomes_aligned.process_embeddings') as mock_process_embeddings, \
             patch('prediction.predict_binary_outcomes_aligned.add_diagnosis_codes') as mock_add_diagnosis_codes, \
             patch('utils.predict_utils.build_subsets') as mock_build_subsets, \
             patch('prediction.predict_binary_outcomes_aligned.perform_cross_validation') as mock_perform_cv, \
             patch('utils.predict_utils.format_results') as mock_format_results, \
             patch('utils.predict_utils.save_results') as mock_save_results, \
             patch('utils.predict_utils.threshold_label', return_value=self.sample_label), \
             patch('utils.predict_utils.verify_data_alignment'), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump'):
                
            # Configure mock returns
            mock_args.return_value = MagicMock(no_cv=False, high=True)
            mock_load_config.return_value = MagicMock(
                tasks=['label'], 
                windows=['two_year'], 
                stratified=True, 
                n_splits=5, 
                seed=42,
                test_size=100,
                label_path_nt=Path('test_path'),
                label_path_linux=Path('test_path'),
                run_label='test_run'
            )
            mock_get_models.return_value = [('TestModel', MagicMock(), {})]
            mock_scan_parquet.return_value = self.sample_baseline.lazy()
            mock_process_embeddings.return_value = {
                'baseline': self.sample_baseline.with_columns(pl.lit(0).alias('label'))
            }
            mock_add_diagnosis_codes.return_value = self.sample_baseline.with_columns(pl.lit(0).alias('label'))
            mock_build_subsets.return_value = self.subset_data
            mock_perform_cv.return_value = [{'subset': 'full_test', 'task': 'label', 'feature_set': 'baseline', 'model': 'TestModel', 'auc': 0.8}]
            mock_format_results.return_value = pd.DataFrame({
                'subset': ['full_test'], 
                'task': ['label'], 
                'feature_set': ['baseline'], 
                'model': ['TestModel'],
                'auc': ['0.800*']
            })
            
            # Call the main function
            pred.main()
            
            # Verify that the function progressed through all steps
            mock_load_config.assert_called_once()
            mock_get_models.assert_called_once()
            mock_scan_parquet.assert_called()
            mock_process_embeddings.assert_called_once()
            mock_add_diagnosis_codes.assert_called_once()
            mock_build_subsets.assert_called_once()
            mock_perform_cv.assert_called_once()
            mock_format_results.assert_called_once()
            mock_save_results.assert_called()


class TestMetrics(unittest.TestCase):
    def test_calculate_metrics(self):
        # Test the calculate_metrics function
        y_test = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        result = calculate_metrics(y_test, y_pred, threshold=0.5)
        
        # Check if all expected metrics are present
        expected_metrics = ['auc', 'ap', 'accuracy', 'recall', 'specificity', 'brier', 'dcal', 'dcal_pvalue']
        for metric in expected_metrics:
            self.assertIn(metric, result)
        
        # Test with homogeneous y_test
        y_test_homogeneous = np.zeros(5)
        result_homogeneous = calculate_metrics(y_test_homogeneous, y_pred, threshold=0.5)
        self.assertIsNone(result_homogeneous)

    def test_d_calibration(self):
        # Test the d_calibration function
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        chi_squared, p_value = d_calibration(y_true, y_pred, n_bins=2)
        
        # Basic checks - we're not testing correctness, just that it returns values
        self.assertIsInstance(chi_squared, float)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(chi_squared, 0)
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    def test_specificity_score(self):
        # Test the specificity_score function
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        specificity = specificity_score(y_true, y_pred)
        
        # For this data, 2 true negatives, 1 false positive
        expected_specificity = 2/3
        self.assertAlmostEqual(specificity, expected_specificity)


class TestPredictUtils(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_df = pl.DataFrame({
            'ppn_int': list(range(1, 101)),
            'cost': np.random.rand(100) * 1000,
            'other_task': np.random.rand(100)
        })
        
        self.sample_baseline = pl.DataFrame({
            'ppn_int': list(range(1, 101)),
            'feature_age_recode': np.random.randint(50, 90, 100),
            'feature_sex_1': np.random.randint(0, 2, 100),
            'feature_sex_2': np.random.randint(0, 2, 100),
            'feature_num_episodes': np.random.randint(1, 10, 100),
            'diagnosis_history': [[f"A{i}", f"B{i}", f"I15.{i%2}", f"E11.{i%3}"] for i in range(100)]
        })
        
        # Aligned dataframes for testing verify_data_alignment
        self.embedding_set = {
            'embed1': pl.DataFrame({'ppn_int': list(range(1, 11)), 'feature1': np.random.rand(10)}),
            'embed2': pl.DataFrame({'ppn_int': list(range(1, 11)), 'feature2': np.random.rand(10)})
        }
        self.label_df = pl.DataFrame({'ppn_int': list(range(1, 11)), 'label': np.random.randint(0, 2, 10)})

    def test_threshold_label(self):
        # Test the threshold_label function for cost task with high=True
        result_high = threshold_label(self.sample_df, 'cost', high=True)
        
        # Check if the function correctly thresholded the values
        threshold = self.sample_df.select(pl.col('cost').quantile(.90)).item()
        expected_label = (self.sample_df['cost'] >= threshold)
        
        # Check if the raw column was added
        self.assertIn('cost_raw', result_high.columns)
        
        # Check if the thresholded label matches expectations
        np.testing.assert_array_equal(result_high['cost'].to_numpy(), expected_label.to_numpy())
        
        # Test with high=False for other task
        result_low = threshold_label(self.sample_df, 'other_task', high=False)
        
        # Check if the function correctly thresholded the values for binary case
        expected_label_low = (self.sample_df['other_task'] > 0)
        
        # Check if the thresholded label matches expectations
        np.testing.assert_array_equal(result_low['other_task'].to_numpy(), expected_label_low.to_numpy())

    def test_load_config(self):
        # Test the load_config function
        with patch('builtins.open', mock_open(read_data='tasks: [label]\nwindows: [two_year]')), \
             patch('yaml.safe_load', return_value={'tasks': ['label'], 'windows': ['two_year']}):
                
            config = load_config('test_config.yaml')
            
            # Check if the config has the expected structure
            self.assertIn('tasks', config)
            self.assertEqual(config.tasks, ['label'])
            self.assertIn('windows', config)
            self.assertEqual(config.windows, ['two_year'])

    def test_build_subsets(self):
        # Test the build_subsets function
        subsets = build_subsets(self.sample_baseline, label='label')
        
        # Check if all expected subsets are present
        expected_subsets = ['over_75', 'under_75', 'female', 'male', 'num_episodes=1', 
                           'num_episodes=2-4', 'num_episodes>=5', 'cvd', 'diabetes', 
                           'ckd', 'mental_health']
        for subset in expected_subsets:
            self.assertIn(subset, subsets)
            self.assertIsInstance(subsets[subset], np.ndarray)
            self.assertEqual(len(subsets[subset]), 100)  # Match our sample size

    def test_verify_data_alignment(self):
        # Test the verify_data_alignment function with aligned data
        # This should not raise any exceptions
        verify_data_alignment(self.embedding_set, self.label_df)
        
        # Test with misaligned data
        misaligned_embed = {
            'embed1': pl.DataFrame({'ppn_int': list(range(1, 11)), 'feature1': np.random.rand(10)}),
            'embed2': pl.DataFrame({'ppn_int': list(range(2, 12)), 'feature2': np.random.rand(10)})  # Offset by 1
        }
        
        with self.assertRaises(AssertionError):
            verify_data_alignment(misaligned_embed, self.label_df)
        
        # Test with different sizes
        different_size_embed = {
            'embed1': pl.DataFrame({'ppn_int': list(range(1, 11)), 'feature1': np.random.rand(10)}),
            'embed2': pl.DataFrame({'ppn_int': list(range(1, 12)), 'feature2': np.random.rand(11)})  # One more row
        }
        
        with self.assertRaises(AssertionError):
            verify_data_alignment(different_size_embed, self.label_df)


if __name__ == '__main__':
    unittest.main()