import os
import numpy as np
from tensorflow import keras
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")

print(f"Checking for model at {MODEL_PATH}...")
model = None

if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Running in CI/mock mode.")

def predict_image(pil_image):
    if model is None:
        return 0.1234 
    
    picture = pil_image.convert("RGB").resize((224, 224))
    image_array = np.array(picture).astype("float32")
    image_batch = np.expand_dims(image_array, axis=0)
    probability = model.predict(image_batch, verbose=0)[0][0]
    return float(probability)
