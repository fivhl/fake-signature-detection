from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog

app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "This endpoint only accepts POST requests with a file."

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # ادامه‌ی کد برای پیش‌بینی...

# Load the model, scaler, and PCA
model, scaler, pca = joblib.load("model2.pkl")

app = Flask(__name__)

def preprocess_image(image_bytes):
    img_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img

def extract_hog_feature(img):
    return hog(img, orientations=18, pixels_per_cell=(3, 3), cells_per_block=(2, 2), block_norm="L2-Hys")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = preprocess_image(file.read())
    feature = extract_hog_feature(img).reshape(1, -1)
    feature_pca = pca.transform(scaler.transform(feature))
    pred = model.predict(feature_pca)[0]

    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")