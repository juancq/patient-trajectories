# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
#import resource
#max_memory = int(1 * 1024 * 1024 *1024)
#resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
import sys
import time
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import miceforest as mf
import unittest
from unittest.mock import patch, MagicMock

# Assuming impute_cost.py is in the same directory
from downstream_prediction.impute_cost import (
    group_other_categories,
    preprocess_data,
    get_cpi,
    impute_data,
    get_cost_data,
    apdc_eddc_impute
)

# If impute_cost.py is in a parent directory, you might need to add it to the Python path
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from downstream_prediction.impute_cost import ...

class TestImputeCost(unittest.TestCase):

    def setUp(self):
        self.test_df = pl.DataFrame({
            'ppn_int': [1, 2, 3],
            'apdc_project_recnum': ['A1', 'A2', 'A3'],
            'sex': ['M', 'F', 'M'],
            'age_recode': [30, 40, 50],
            'episode_length_of_stay': [2, 3, 4],
            'block_nump': ['B1', 'B2', 'B3'],
            'diagnosis_codep': ['D1', 'D2', 'D3'],
            'indigenous_status': ['1', '2', '3'],
            'emergency_status_recode': ['E1', 'E2', 'E3'],
            'episode_start_date': ['2020-01-01', '2020-02-01', '2020-03-01']
        })

    def test_group_other_categories(self):
        df = group_other_categories(self.test_df.lazy(), ['sex'], threshold=2)
        df = df.collect()
        self.assertEqual(df.select('sex').to_dict()['sex'].to_list(), ['M', 'other', 'M'])

        df = group_other_categories(self.test_df.lazy(), ['sex'], threshold=0)
        df = df.collect()
        self.assertEqual(df.select('sex').to_dict()['sex'].to_list(), ['M', 'F', 'M'])

    def test_preprocess_data_apdc(self):
        df = preprocess_data('apdc', self.test_df.lazy(), ['sex', 'block_nump'])
        df = df.collect()
        self.assertIsInstance(df.select('sex').to_series(), pl.Series)
        self.assertIsInstance(df.select('block_nump').to_series(), pl.Series)

    @patch('miceforest.ImputationKernel')
    def test_impute_data(self, mock_imputation_kernel):
        mock_kernel = MagicMock()
        mock_imputation_kernel.return_value = mock_kernel
        mock_kernel.complete_data.return_value = pd.DataFrame({'total': [100, 200, 300]})
        
        data = pd.DataFrame({'total': [100, None, 300]})
        variable_schema = {'total': ['age']}
        result = impute_data(data, variable_schema, 'total', 'test.png')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_get_cost_data(self):
        with patch('polars.scan_parquet') as mock_scan:
            mock_scan.return_value = pl.DataFrame({
                'apdc_project_recnum': ['A1', 'A2', 'A3'],
                'total': [100, 200, 300]
            }).lazy()
            result = get_cost_data('test.parquet', 'apdc_project_recnum', 'total')
            self.assertIsInstance(result, pl.LazyFrame)
            self.assertEqual(result.collect().shape, (3, 2))

    @patch('downstream_prediction.impute_cost.get_cost_data')
    @patch('downstream_prediction.impute_cost.impute_data')
    def test_apdc_eddc_impute(self, mock_impute_data, mock_get_cost_data):
        mock_get_cost_data.return_value = pl.DataFrame({
            'apdc_project_recnum': ['A1', 'A2', 'A3'],
            'total': [100, 200, 300]
        }).lazy()
        mock_impute_data.return_value = pd.DataFrame({
            'total': [150, 250, 350]
        })
        
        result = apdc_eddc_impute('apdc', self.test_df.lazy(), Path('test'))
        
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result.shape, (3, 3))  # ppn_int, apdc_project_recnum, total

if __name__ == '__main__':
    unittest.main()
