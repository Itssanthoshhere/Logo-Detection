# app_tflite.py
from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from PIL import Image
import logging
import tensorflow as tf

MODEL_PATH = "logo_int8.tflite"  # or "logo.tflite"
CLASSES_PATH = "classes.txt"
IMG_SIZE = (224, 224)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logo_api")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Running on Render?
RUNNING_ON_RENDER = os.environ.get("RENDER", None) == "true"
UPLOAD_FOLDER = "/tmp/uploads" if RUNNING_ON_RENDER else "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load classes
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# Load TFLite model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at {MODEL_PATH}")

# Interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input details
input_index = input_details[0]['index']
input_dtype = input_details[0]['dtype']
input_shape = input_details[0]['shape']  # e.g. [1,224,224,3]

# Output details
output_index = output_details[0]['index']
output_dtype = output_details[0]['dtype']

logger.info("Loaded TFLite model: %s", MODEL_PATH)
logger.info("Input details: %s", input_details)
logger.info("Output details: %s", output_details)

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # normalized float
    if input_dtype == np.int8:
        # quantized model expected int8 inputs
        # get quantization parameters (scale, zero_point)
        scale, zero_point = input_details[0]['quantization']
        if scale == 0:
            raise ValueError("Quantization scale is zero")
        arr = arr / scale + zero_point
        arr = np.round(arr).astype(np.int8)
    else:
        # float model: ensure correct dtype
        arr = arr.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tflite(image_path):
    inp = preprocess_image(image_path)
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)

    # if output is quantized
    if output_dtype == np.int8:
        scale, zero_point = output_details[0]['quantization']
        # dequantize
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    probs = np.squeeze(output_data)
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_idx, confidence, probs

@app.route("/uploaded/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

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
        pred_idx, confidence, _ = predict_tflite(save_path)
        label = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
        return render_template("index.html",
                               prediction=label,
                               confidence=round(confidence * 100, 2),
                               image_path=f"/uploaded/{file.filename}")
    except Exception as e:
        logger.exception("Prediction error")
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
