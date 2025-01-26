import os
from tensorflow.keras.datasets import cifar10
from utils.data_utils import save_data, preprocess_data

# Define file paths
RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")

def preprocess_and_save_data():
    """Downloads CIFAR-10, preprocesses it, and saves raw/processed data."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("Raw data downloaded.")

    # Save raw data
    save_data((x_train, y_train), os.path.join(RAW_DIR, "train_data.pkl"))
    save_data((x_test, y_test), os.path.join(RAW_DIR, "test_data.pkl"))
    print("Raw data saved to raw/.")

    # Preprocess the data
    x_train, x_test = preprocess_data(x_train, x_test)

    # Save processed data
    save_data((x_train, y_train), os.path.join(PROCESSED_DIR, "train_data.pkl"))
    save_data((x_test, y_test), os.path.join(PROCESSED_DIR, "test_data.pkl"))
    print("Processed data saved to processed/.")

if __name__ == "__main__":
    preprocess_and_save_data()
