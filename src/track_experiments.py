import os
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_utils import load_data

# Define absolute path for processed data
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Navigate to project root
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
train_data_path = os.path.join(PROCESSED_DIR, "train_data.pkl")
test_data_path = os.path.join(PROCESSED_DIR, "test_data.pkl")

# Load processed data
train_data = load_data(train_data_path)
test_data = load_data(test_data_path)
train_x, train_y = train_data
test_x, test_y = test_data

# Get subset of data
SUBSET_SIZE = 1000  # Adjust this value based on your needs
train_x, train_y = train_x[:SUBSET_SIZE], train_y[:SUBSET_SIZE]
test_x, test_y = test_x[:SUBSET_SIZE//2], test_y[:SUBSET_SIZE//2]

# MLflow experiment setup
mlflow.set_experiment("CIFAR-10 SVM Experiment")

# Define parameter sets to test
param_sets = [
    {"kernel": "linear", "C": 1},
    {"kernel": "rbf", "C": 1, "gamma": "scale"},
    {"kernel": "rbf", "C": 10, "gamma": "auto"},
]

for params in param_sets:
    with mlflow.start_run():
        # Train the model
        model = SVC(**params)
        model.fit(train_x, train_y.ravel())

        # Evaluate the model
        y_pred = model.predict(test_x)
        accuracy = accuracy_score(test_y, y_pred)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Provide an input example for schema inference
        input_example = train_x[0].reshape(1, -1)  # Single input example
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            input_example=input_example
        )

        print(f"Run completed with params: {params}, accuracy: {accuracy:.2f}")

