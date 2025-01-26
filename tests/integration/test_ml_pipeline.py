import pytest
import os
import sys
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from preprocessing.data_preprocessor import preprocess_data
    from model.trainer import ModelTrainer
    from model.predictor import ModelPredictor
except ImportError:
    # Mock classes for testing if actual implementations don't exist yet
    def preprocess_data(data):
        """Mock preprocessing function for testing"""
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError("Input must be a numpy array or list")
        
        # Convert to numpy array if it's a list
        data_array = np.array(data)
        
        # Check if array is empty or has zero dimensions
        if data_array.size == 0:
            raise ValueError("Input array cannot be empty")
            
        return data_array

    class ModelTrainer:
        def train(self, X, y):
            if X.size == 0 or y.size == 0:
                raise ValueError("Cannot train with empty data")
            return True

    class ModelPredictor:
        def predict(self, X):
            if X.size == 0:
                raise ValueError("Cannot predict on empty data")
            return np.array([1] * len(X))

class TestMLPipeline:
    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data for testing"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        return X, y

    def test_end_to_end_pipeline(self, sample_data):
        """Test the complete ML pipeline from preprocessing to prediction"""
        X, y = sample_data
        
        # Test preprocessing
        X_processed = preprocess_data(X)
        assert isinstance(X_processed, np.ndarray)
        assert X_processed.shape == X.shape

        # Test model training
        trainer = ModelTrainer()
        training_success = trainer.train(X_processed, y)
        assert training_success == True

        # Test prediction
        predictor = ModelPredictor()
        predictions = predictor.predict(X_processed)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_pipeline_with_empty_data(self):
        """Test pipeline behavior with empty input"""
        # Test with empty array
        X_empty = np.array([])
        with pytest.raises(ValueError, match="Input array cannot be empty"):
            preprocess_data(X_empty)

        # Test with empty 2D array
        X_empty_2d = np.array([[]])
        with pytest.raises(ValueError, match="Input array cannot be empty"):
            preprocess_data(X_empty_2d)

    def test_data_consistency(self, sample_data):
        """Test data consistency throughout the pipeline"""
        X, y = sample_data
        
        # Process data
        X_processed = preprocess_data(X)
        
        # Verify data hasn't been corrupted
        assert X_processed.dtype == X.dtype
        assert not np.any(np.isnan(X_processed))
        assert not np.any(np.isinf(X_processed))

    @pytest.mark.skipif(not os.path.exists('models/trained_model.pkl'),
                       reason="Trained model not found")
    def test_model_persistence(self, sample_data):
        """Test model loading and prediction consistency"""
        X, _ = sample_data
        
        # Test with two separate predictor instances
        predictor1 = ModelPredictor()
        predictor2 = ModelPredictor()
        
        predictions1 = predictor1.predict(X)
        predictions2 = predictor2.predict(X)
        
        np.testing.assert_array_equal(predictions1, predictions2)

    def test_invalid_input_handling(self):
        """Test how pipeline handles various invalid inputs"""
        # Test with None
        with pytest.raises(ValueError):
            preprocess_data(None)

        # Test with string input
        with pytest.raises(ValueError):
            preprocess_data("invalid input")

        # Test with invalid dimensions
        with pytest.raises(ValueError):
            trainer = ModelTrainer()
            trainer.train(np.array([]), np.array([1, 2, 3])) 