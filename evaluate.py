import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay

# 1. Find files exactly like we did in data_pipeline
ai_folder = "data/AiArtData/AiArtData"
real_folder = "data/RealArt/RealArt"

paths = []
labels = []
for file in os.listdir(ai_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        paths.append(os.path.join(ai_folder, file))
        labels.append(1)
for file in os.listdir(real_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        paths.append(os.path.join(real_folder, file))
        labels.append(0)

combined = list(zip(paths, labels))
random.seed(42)
random.shuffle(combined)
paths, labels = zip(*combined)
paths = list(paths)
labels = list(labels)

val_end = int(len(paths) * 0.85)
val_paths = paths[int(len(paths)*0.70):val_end]
val_labels = labels[int(len(paths)*0.70):val_end]
test_paths = paths[val_end:]
test_labels = labels[val_end:]

def load_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    return img, tf.cast(label, tf.float32)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).map(load_and_resize).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels)).map(load_and_resize).batch(32)

class_folders = ["RealArt", "AiArtData"]
model_names = ["baseline", "transfer"]
model_files = ["Artifacts/baseline.keras", "Artifacts/transfer.keras"]

rows = []
saved_probabilities = []
thresholds = []

for number in range(2):
    model = keras.models.load_model(model_files[number])
    val_probs = model.predict(val_ds).flatten()

    best_threshold = 0.5
    best_recall = 0
    for step in range(30, 85, 5):
        threshold = step / 100
        preds = (val_probs >= threshold).astype(int)
        prec = precision_score(val_labels, preds, zero_division=0)
        rec = recall_score(val_labels, preds, zero_division=0)
        if prec >= 0.90 and rec > best_recall:
            best_recall = rec
            best_threshold = threshold

    thresholds.append(best_threshold)
    probs = model.predict(test_ds).flatten()
    saved_probabilities.append(probs)
    preds = (probs >= best_threshold).astype(int)

    rows.append({
        "model": model_names[number], "threshold": best_threshold,
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, zero_division=0),
        "recall": recall_score(test_labels, preds, zero_division=0),
        "f1": f1_score(test_labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, probs)
    })

    matrix = confusion_matrix(test_labels, preds)
    ConfusionMatrixDisplay(matrix, display_labels=class_folders).plot(colorbar=False)
    plt.title(model_names[number])
    plt.savefig("Artifacts/cm_" + model_names[number] + ".png")
    plt.close()

table = pd.DataFrame(rows)
table.to_csv("Artifacts/test_metrics.csv", index=False)
print(table)

plt.figure(figsize=(6, 5))
for number in range(2):
    fpr, tpr, points = roc_curve(test_labels, saved_probabilities[number])
    plt.plot(fpr, tpr, label=model_names[number])
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate"); plt.ylabel("true positive rate"); plt.legend()
plt.savefig("Artifacts/roc_curves.png"); plt.close()

better = 0 if rows[0]["roc_auc"] > rows[1]["roc_auc"] else 1
probs = saved_probabilities[better]
preds = (probs >= thresholds[better]).astype(int)

wrong_numbers = [i for i in range(len(test_labels)) if preds[i] != test_labels[i]]
print("total mistakes", len(wrong_numbers))

plt.figure(figsize=(15, 6))
for place in range(min(10, len(wrong_numbers))):
    num = wrong_numbers[place]
    picture = Image.open(test_paths[num]).convert("RGB")
    plt.subplot(2, 5, place + 1)
    plt.imshow(picture)
    plt.title("true: " + class_folders[test_labels[num]] + "\nP(AI)=" + str(round(float(probs[num]), 2)), fontsize=8)
    plt.axis("off")
plt.savefig("Artifacts/error_grid.png"); plt.close()

error_paths = [test_paths[wrong_numbers[i]] for i in range(min(10, len(wrong_numbers)))]
pd.DataFrame({"path": error_paths}).to_csv("Artifacts/error_cases.csv", index=False)