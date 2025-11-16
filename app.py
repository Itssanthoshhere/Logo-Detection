from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import logging

MODEL_PATH = "logo.h5"
CLASSES_PATH = "classes.txt"
IMG_SIZE = (224, 224)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logo_api")

app = Flask(__name__, static_folder="static", template_folder="templates")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your logo.h5 in the project folder.")

model = load_model(MODEL_PATH)
logger.info("Model loaded: %s", MODEL_PATH)

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file not found at {CLASSES_PATH}. Create a classes.txt with one class per line.")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

logger.info("Loaded %d classes", len(class_names))

def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.array(preds)

    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds.reshape(preds.shape[0], preds.shape[1])
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    raw_vector = preds[0]
    pred_index = int(np.argmax(raw_vector))
    confidence = float(np.max(raw_vector))

    return pred_index, confidence, raw_vector

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")

    save_dir = "static/uploads"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)

    try:
        pred_index, confidence, raw_vector = predict_image(save_path)
        if pred_index >= len(class_names):
            pred_label = "Unknown (index out of range)"
        else:
            pred_label = class_names[pred_index]

        return render_template(
            "index.html",
            prediction=pred_label,
            confidence=round(confidence * 100, 2),
            image_path=save_path
        )
    except Exception as exc:
        logger.exception("Prediction error")
        return render_template("index.html", error=str(exc))

if __name__ == "__main__":
    app.run(debug=True)
