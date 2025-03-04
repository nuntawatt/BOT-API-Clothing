import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# โหลดโมเดลที่ฝึกไว้
MODEL_PATH = "trained_voting_classifierV3.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH else None

# แมปค่าขนาดเสื้อ
SIZE_MAP = {
    0: "XS",
    1: "S",
    2: "M",
    3: "L",
    4: "XL",
    5: "XXL"
}

# ฟีเจอร์ที่ใช้ในการพยากรณ์
FEATURES = ["age", "height", "weight"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API for size prediction is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        # ตรวจสอบคีย์ที่จำเป็น
        if not all(key in data for key in FEATURES):
            return jsonify({"error": "Missing required fields"}), 400

        # ดึงข้อมูลและแปลงเป็น float
        input_values = np.array([[float(data["age"]), float(data["height"]), float(data["weight"])]])
        print(f"Features: {input_values}")

        # ตรวจสอบว่าโมเดลโหลดสำเร็จ
        if model is None:
            return jsonify({"error": "Model not found"}), 500

        # พยากรณ์ขนาดเสื้อ
        prediction = model.predict(input_values)[0]
        size_prediction = SIZE_MAP.get(prediction, "Unknown Size")

        print(f"Prediction: {size_prediction}")
        return jsonify({"prediction": size_prediction})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
