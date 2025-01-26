from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
MODEL_PATH = "models/tuned/best_svm_model.pkl"
model = joblib.load(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "SVM Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
