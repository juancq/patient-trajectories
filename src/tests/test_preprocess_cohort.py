import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pipeline.preprocess_cohort import main

@pytest.fixture
def mock_data():
    apdc_data = pl.DataFrame({
        'ppn_int': [1, 1, 2],
        'hospital_type': [1, 2, 3],
        'peer_group': [None, 'A', 'B'],
        'facility_identifier_e': ['F1', 'F2', 'F3'],
        'episode_start_date': ['01/01/2023', '02/01/2023', '03/01/2023'],
        'episode_end_date': ['01/02/2023', '02/02/2023', '03/02/2023'],
        'episode_start_time': ['10:00', '11:00', '12:00'],
        'episode_end_time': ['14:00', '15:00', '16:00'],
        'age_recode': [30, None, 40],
        'sex': ['M', None, 'F'],
        'mthyr_birth': ['1990-01', None, '1980-01'],
        'block_num_1': ['A1', 'B1', 'C1'],
        'block_num_2': ['A2', None, 'C2'],
    })
    
    eddc_data = pl.DataFrame({
        'ppn_int': [3],
        'arrival_date': ['04/01/2023'],
        'actual_departure_date': ['04/02/2023'],
        'arrival_time': ['13:00'],
        'actual_departure_time': ['17:00'],
        'mthyr_birth_date': ['1970-01'],
    })
    
    return apdc_data, eddc_data

@pytest.fixture
def mock_paths(tmp_path):
    apdc_eddc_path = tmp_path / 'study1'
    apdc_eddc_path.mkdir(parents=True)
    preprocessed_path = apdc_eddc_path / 'preprocessed'
    preprocessed_path.mkdir()
    return apdc_eddc_path, preprocessed_path

def test_main(mock_data, mock_paths, monkeypatch):
    apdc_data, eddc_data = mock_data
    apdc_eddc_path, preprocessed_path = mock_paths
    
    # Mock the scan_parquet function
    def mock_scan_parquet(path):
        if 'apdc' in str(path):
            return pl.LazyFrame(apdc_data)
        elif 'eddc' in str(path):
            return pl.LazyFrame(eddc_data)
    
    monkeypatch.setattr(pl, 'scan_parquet', mock_scan_parquet)
    
    # Mock the Path
    monkeypatch.setattr(Path, '__truediv__', lambda self, other: mock_paths[0] / other)
    
    # Run the main function
    with patch('pipeline.preprocess_cohort.track', lambda x: x):
        main()
    
    # Check if the output file was created
    assert (preprocessed_path / 'group_0.parquet').exists()
    
    # Read the output file and perform assertions
    result = pl.read_parquet(preprocessed_path / 'group_0.parquet')
    
    # Test APDC specific preprocessing
    assert result.filter(pl.col('apdc') == 1)['peer_group'].to_list() == [None, 'A', 'PRIV']
    assert result.filter(pl.col('apdc') == 1)['facility_identifier'].to_list() == ['F1', 'F2', 'F3']
    assert result.filter(pl.col('apdc') == 1)['hospital_type'].to_list() == [False, True, True]
    
    # Test EDDC specific preprocessing
    assert result.filter(pl.col('eddc') == 1)['episode_start_date'].to_list() == [pl.Date(2023, 4, 1)]
    assert result.filter(pl.col('eddc') == 1)['episode_end_date'].to_list() == [pl.Date(2023, 4, 2)]
    
    # Test generic preprocessing
    assert all(isinstance(date, pl.Date) for date in result['episode_start_date'])
    assert all(isinstance(date, pl.Date) for date in result['episode_end_date'])
    assert all(0 <= time <= 86399 for time in result['episode_start_time'])
    assert all(0 <= time <= 86399 for time in result['episode_end_time'])
    
    # Test imputation
    assert result.filter(pl.col('ppn_int') == 1)['age_recode'].is_null().sum() == 0
    assert result.filter(pl.col('ppn_int') == 1)['sex'].is_null().sum() == 0
    assert result.filter(pl.col('ppn_int') == 1)['mthyr_birth'].is_null().sum() == 0
    
    # Test procedure list
    assert all(isinstance(procs, list) for procs in result['procedure_list'])
    assert result.filter(pl.col('apdc') == 1)['procedure_list'].to_list() == [['A1', 'A2'], ['B1'], ['C1', 'C2']]

if __name__ == '__main__':
    pytest.main()

