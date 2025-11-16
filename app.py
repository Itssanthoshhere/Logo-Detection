from flask import Flask, render_template, request, send_from_directory
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

# -------- Detect if running on Render (uses /tmp) -------- #
RUNNING_ON_RENDER = os.environ.get("RENDER", None) == "true"

if RUNNING_ON_RENDER:
    UPLOAD_FOLDER = "/tmp/uploads"
else:
    UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logger.info(f"UPLOAD_FOLDER set to: {UPLOAD_FOLDER}")


# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# -----------------------------
# Load Classes
# -----------------------------
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")

with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return pred_index, confidence


# -----------------------------
# Serve Uploaded Images
# -----------------------------
@app.route("/uploaded/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# -----------------------------
# Routes
# -----------------------------
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

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        pred_index, confidence = predict_image(save_path)
        label = class_names[pred_index]

        return render_template(
            "index.html",
            prediction=label,
            confidence=round(confidence * 100, 2),
            image_path=f"/uploaded/{file.filename}"   # WORKS BOTH LOCAL + RENDER
        )
    except Exception as e:
        logger.exception("Prediction error")
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
