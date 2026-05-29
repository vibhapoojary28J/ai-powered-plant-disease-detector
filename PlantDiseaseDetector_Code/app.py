from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# ---------- CONFIG ----------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Filenames (must exist in project folder)
MODEL_PKL = "model.pkl"            # SVM model saved with joblib
LABEL_ENCODER_PKL = "label_encoder.pkl"
SCALER_PKL = "scaler.pkl"
PCA_PKL = "pca.pkl"

# ---------- GLOBALS (will be loaded at startup) ----------
model = None
label_encoder = None
scaler = None
pca = None

# ---------- UTILS: load everything ----------
def load_artifacts():
    global model, label_encoder, scaler, pca

    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(f"{MODEL_PKL} not found. Train the model and place it here.")

    model = joblib.load(MODEL_PKL)
    print("Loaded model:", MODEL_PKL)

    if os.path.exists(LABEL_ENCODER_PKL):
        label_encoder = joblib.load(LABEL_ENCODER_PKL)
        print("Loaded label encoder.")
    else:
        label_encoder = None
        print("Warning: label_encoder.pkl not found - will return numeric class index.")

    if os.path.exists(SCALER_PKL):
        scaler = joblib.load(SCALER_PKL)
        print("Loaded scaler.")
    else:
        scaler = None
        print("Warning: scaler.pkl not found - predictions may be wrong.")

    if os.path.exists(PCA_PKL):
        pca = joblib.load(PCA_PKL)
        print("Loaded PCA.")
    else:
        pca = None
        print("Warning: pca.pkl not found - predictions may be wrong.")


# ---------- Preprocessing (must match training) ----------
def preprocess_image_for_svm(img_path, target_size=(20, 20)):
    """
    1) load image resized to target_size
    2) convert to array, flatten
    3) apply scaler and PCA (if available)
    Returns: 2D-array shaped (1, n_features)
    """
    img = load_img(img_path, target_size=target_size)
    arr = img_to_array(img).flatten().reshape(1, -1).astype("float32")
    # normalize pixel range same as training (if used)
    # Note: scaler should already have been fit on raw flattened pixels (0-255)
    if scaler is not None:
        arr = scaler.transform(arr)
    else:
        arr = arr / 255.0  # fallback normalization

    if pca is not None:
        arr = pca.transform(arr)

    return arr


# ---------- Prediction function ----------
def predict_disease(img_path):
    if model is None:
        raise RuntimeError("Model not loaded")

    x = preprocess_image_for_svm(img_path)

    # If model supports predict_proba (SVM with probability=True), give probability too
    try:
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
        pred_idx = int(model.predict(x)[0])
    except Exception as e:
        # fallback: if model expects different shape / type, raise
        raise

    if label_encoder is not None:
        try:
            label = label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            label = str(pred_idx)
    else:
        label = str(pred_idx)

    result = {"label": label}
    if probs is not None:
        # return top-3 probabilities (label + prob)
        top_k = 3
        indices = np.argsort(probs)[::-1][:top_k]
        top = []
        for i in indices:
            name = label_encoder.inverse_transform([i])[0] if label_encoder is not None else str(i)
            top.append({"label": name, "probability": float(probs[i])})
        result["top_predictions"] = top

    return result


# ---------- Recommendations mapping ----------
recommendations = {
    "Apple___Apple_scab": "Remove infected leaves. Use fungicides like captan or myclobutanil. Ensure proper pruning for airflow.",
    "Apple___Black_rot": "Remove mummified fruit and prune infected twigs. Apply preventive fungicide sprays in spring.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees if possible. Use fungicides (myclobutanil, propiconazole).",
    "Apple___healthy": "The plant is healthy. Continue regular watering and apply preventive fungicide once a year.",

    "Blueberry___healthy": "The plant is healthy. Maintain soil pH 4.5–5.5, avoid overwatering.",

    "Cherry_(including_sour)___healthy": "The plant is healthy. Maintain good airflow and prune regularly.",
    "Cherry_(including_sour)___Powdery_mildew": "Remove infected leaves. Use sulfur-based fungicides or potassium bicarbonate.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant varieties. Apply fungicides (strobilurins or triazoles). Rotate crops.",
    "Corn_(maize)___Common_rust_": "Plant resistant hybrids. Apply fungicides if infection is severe.",
    "Corn_(maize)___healthy": "The plant is healthy. Maintain nitrogen supply and proper spacing.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant seeds. Apply fungicides at tasseling stage.",

    "Grape___Black_rot": "Remove infected leaves and berries. Apply fungicides like mancozeb or myclobutanil. Improve air circulation.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood. Avoid injuries. Apply trunk-protective fungicides.",
    "Grape___healthy": "The plant is healthy. Ensure proper trellising and sunlight.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove diseased leaves. Spray copper-based fungicides.",

    "Orange___Haunglongbing_(Citrus_greening)": "Remove and destroy infected trees. Control psyllids using insecticides. No cure exists.",

    "Peach___Bacterial_spot": "Remove infected foliage. Use copper sprays. Improve drainage.",
    "Peach___healthy": "The plant is healthy. Maintain regular fertilization.",

    "Pepper,_bell___Bacterial_spot": "Use disease-free seeds. Apply copper-based bactericides.",
    "Pepper,_bell___healthy": "The plant is healthy. Avoid overhead watering and provide full sunlight.",

    "Potato___Early_blight": "Remove infected leaves. Use chlorothalonil or mancozeb sprays.",
    "Potato___healthy": "The plant is healthy. Ensure proper soil moisture.",
    "Potato___Late_blight": "Destroy infected plants. Apply fungicides like cyazofamid or mancozeb.",

    "Raspberry___healthy": "The plant is healthy. Maintain airflow and remove old canes.",

    "Soybean___healthy": "The plant is healthy. Ensure weed control and proper nutrient balance.",

    "Squash___Powdery_mildew": "Use sulfur fungicides or neem oil. Improve ventilation.",

    "Strawberry___healthy": "The plant is healthy. Avoid waterlogging and remove old leaves.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply copper-based fungicides.",

    "Tomato___Bacterial_spot": "Use copper sprays. Avoid touching plants when wet.",
    "Tomato___Early_blight": "Remove lower infected leaves. Apply chlorothalonil. Use drip irrigation.",
    "Tomato___Late_blight": "Remove infected plants. Use fungicides like mancozeb or metalaxyl.",
    "Tomato___Leaf_Mold": "Increase airflow. Reduce humidity. Apply fungicide if needed.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Use fungicides such as chlorothalonil.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray neem oil or miticides. Increase humidity to reduce mite spread.",
    "Tomato___Target_Spot": "Use copper sprays. Improve airflow and avoid overhead watering.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies. Remove infected plants. No direct cure.",
}


def get_recommendation(disease_name):
    return recommendations.get(disease_name, "No recommendation available — consult an expert.")



# ---------- Routes ----------
@app.route("/")
def home():
    # `index.html` should be inside templates/
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        pred = predict_disease(filepath)
        disease = pred["label"]
        recommendation = get_recommendation(disease)

        response = {
            "disease": disease,
            "recommendation": recommendation
        }

        # include top predictions if available
        if "top_predictions" in pred:
            response["top_predictions"] = pred["top_predictions"]

    except Exception as e:
        print("\nPREDICTION ERROR:\n", e)
        return jsonify({"error": str(e)}), 500

    return jsonify(response)


# ---------- Main ----------
if __name__ == "__main__":
    try:
        load_artifacts()
    except Exception as e:
        print("Startup error:", e)
        raise

    print("AI Plant Doctor (SVM + PCA) is running...")
    app.run(debug=True)
