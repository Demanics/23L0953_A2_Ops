import os
import numpy as np
import mlflow
import mlflow.keras
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

# Load the champion model straight from the registry by its alias
model = mlflow.keras.load_model("models:/VisionGuard@champion")
print("Loaded champion model from registry")

# Pick the first image from the RealArt folder
real_folder = "data/RealArt/RealArt"
files = [f for f in os.listdir(real_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_path = os.path.join(real_folder, files[0])

# Prepare the image
picture = Image.open(image_path).convert("RGB").resize((224, 224))
image_array = np.array(picture).astype("float32")
image_batch = np.expand_dims(image_array, axis=0)

# Classify
probability = model.predict(image_batch, verbose=0)[0][0]
label = "AiArtData (AI)" if probability >= 0.5 else "RealArt (Real)"

print("Image:", image_path)
print("P(AI):", round(float(probability), 4))
print("Prediction:", label)