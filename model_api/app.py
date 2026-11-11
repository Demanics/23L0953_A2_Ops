from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import os

app = Flask(__name__)

print("Loading model...")
model = keras.models.load_model("model.keras")
print("Model loaded successfully!")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img_bytes = file.read()
    
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    arr = np.expand_dims(np.array(img).astype("float32"), axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    
    label = "AiArtData" if prob >= 0.5 else "RealArt"
    conf = prob if label == "AiArtData" else 1.0 - prob
    
    return jsonify({
        "label": label,
        "confidence": round(conf, 4),
        "raw_probability": round(prob, 4)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))