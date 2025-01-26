import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def save_data(data, file_path):
    """Saves data to the specified file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(data, file_path)


def load_data(file_path):
    """Loads data from the specified file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return joblib.load(file_path)


def preprocess_data(x_train, x_test):
    """Flattens and normalizes the data."""
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.astype(np.float32))
    x_test = scaler.transform(x_test.astype(np.float32))

    return x_train, x_test
