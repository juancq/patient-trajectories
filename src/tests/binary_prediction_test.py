#import sys
import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest

from downstream_prediction.predict_binary_outcomes import get_embeddings


def test_get_embeddings_success():
	fname = '/mnt/data_volume/apdc/study1/preprocessed/tasks/six_months_halo_embeddings_last.parquet'
	content = get_embeddings(fname)
	assert content is not None

def test_get_embeddings_invalid_name():
	invalid_path = 'hello_world'
	with pytest.raises(ValueError) as exc_info:
		get_embeddings(invalid_path)
	
	assert str(exc_info.value) == f'Invalid embedding path: {invalid_path}'
	

if __name__ == "__main__":
    pytest.main()
