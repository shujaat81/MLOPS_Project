import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils.data_utils import load_data
from utils.log_utils import save_results

# Load processed data
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
train_data = load_data(os.path.join(PROCESSED_DIR, "train_data.pkl"))
test_data = load_data(os.path.join(PROCESSED_DIR, "test_data.pkl"))
train_x, train_y = train_data
test_x, test_y = test_data

# Get subset of data
SUBSET_SIZE = 1000  # Adjust this value based on your needs
train_x, train_y = train_x[:SUBSET_SIZE], train_y[:SUBSET_SIZE]
test_x, test_y = test_x[:SUBSET_SIZE//2], test_y[:SUBSET_SIZE//2]

# Train and evaluate SVM
svm = SVC(kernel='linear', C=1)
svm.fit(train_x, train_y.ravel())
y_pred = svm.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)

# Log results
print(f"Baseline Accuracy: {accuracy:.2f}")
results = {
    "model": "SVM",
    "kernel": "linear",
    "C": 1,
    "test_accuracy": accuracy
}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../models/trained")
save_results(results, os.path.join(RESULTS_DIR, "baseline_results.json"))
