import os
from inference import predict_image

# Test on one AI image and one Real image
ai_folder = "data/AiArtData/AiArtData"
real_folder = "data/RealArt/RealArt"

# Get first image from each folder
ai_files = [f for f in os.listdir(ai_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
real_files = [f for f in os.listdir(real_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

ai_image_path = os.path.join(ai_folder, ai_files[0])
real_image_path = os.path.join(real_folder, real_files[0])

# Test the AI image
print("\n=== Testing AI Image ===")
print("Image:", ai_image_path)
label, confidence = predict_image(ai_image_path)
print("Predicted:", label)
print("Confidence:", round(confidence, 4))

# Test the Real image
print("\n=== Testing Real Image ===")
print("Image:", real_image_path)
label, confidence = predict_image(real_image_path)
print("Predicted:", label)
print("Confidence:", round(confidence, 4))