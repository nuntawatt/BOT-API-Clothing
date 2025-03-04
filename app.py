import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  

final_model = joblib.load("trained_voting_classifierv1.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoder"]

app = Flask(__name__)
CORS(app)  

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running on Render"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        if "age" not in data or "height" not in data or "weight" not in data:
            return jsonify({"error": "Missing required fields: age, height, weight"}), 400
        
        age = data["age"]
        height = data["height"]
        weight = data["weight"]

        features = [float(age), float(height), float(weight)]
        print(f"Features: {features}")

        features_array = np.array([features]).reshape(1, -1)
        
        probabilities = model.predict_proba(features_array)[0]  
        
        prediction = model.predict(features_array)[0]
        size_predicted = label_encoders.inverse_transform([prediction])[0]
        
        result = {
            "prediction": size_predicted,
            "probabilities": {}
        }
        
        for i, size in enumerate(label_encoders.classes_):
            result["probabilities"][size] = probabilities[i]

        print(f"Prediction: {size_predicted}, Probabilities: {result['probabilities']}")
        return jsonify(result)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
