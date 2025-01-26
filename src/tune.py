import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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

# Define hyperparameter grid
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

# Perform hyperparameter tuning
svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2)
grid_search.fit(train_x, train_y.ravel())

# Get the best parameters and accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_accuracy:.2f}")

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(test_x)
test_accuracy = accuracy_score(test_y, y_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Log results
results = {
    "best_parameters": best_params,
    "best_cross_validation_accuracy": best_accuracy,
    "test_accuracy": test_accuracy
}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../models/tuned")
save_results(results, os.path.join(RESULTS_DIR, "tuning_results.json"))

# Save the best model
joblib.dump(best_model, os.path.join(RESULTS_DIR, "best_svm_model.pkl"))
print("Best model saved.")
