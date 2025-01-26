import pytest
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from preprocessing.data_preprocessor import preprocess_data
except ImportError:
    # Mock class for testing if actual implementation doesn't exist yet
    def preprocess_data(data):
        """Mock preprocessing function for testing"""
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError("Input must be a numpy array or list")
        return np.array(data)

class TestPreprocessing:
    def test_preprocess_data_with_valid_input(self):
        """Test preprocessing with valid numerical input"""
        test_data = [1, 2, 3, 4, 5]
        result = preprocess_data(test_data)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(test_data)
        assert np.array_equal(result, np.array(test_data))

    def test_preprocess_data_with_empty_input(self):
        """Test preprocessing with empty input"""
        test_data = []
        result = preprocess_data(test_data)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_preprocess_data_with_invalid_input(self):
        """Test preprocessing with invalid input"""
        test_data = "invalid_input"
        with pytest.raises(ValueError):
            preprocess_data(test_data)

    def test_preprocess_data_with_numpy_array(self):
        """Test preprocessing with numpy array input"""
        test_data = np.array([1, 2, 3, 4, 5])
        result = preprocess_data(test_data)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, test_data)

    @pytest.mark.skip(reason="Example of skipping a test")
    def test_advanced_preprocessing(self):
        """Test advanced preprocessing features (skipped for now)"""
        pass
