import os
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

os.makedirs("Artifacts", exist_ok=True)

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

print("Total images found:", len(paths))

combined = list(zip(paths, labels))
random.seed(42)
random.shuffle(combined)

paths, labels = zip(*combined)
paths = list(paths)
labels = list(labels)

train_end = int(len(paths) * 0.70)
val_end = int(len(paths) * 0.85)

train_paths = paths[:train_end]
train_labels = labels[:train_end]
val_paths = paths[train_end:val_end]
val_labels = labels[train_end:val_end]
test_paths = paths[val_end:]
test_labels = labels[val_end:]

print("Train:", len(train_paths), " Val:", len(val_paths), " Test:", len(test_paths))

def load_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    return img, tf.cast(label, tf.float32)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(1000).map(load_and_resize).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_resize).batch(32).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_and_resize).batch(32).prefetch(tf.data.AUTOTUNE)

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])

train_ds = train_ds.map(lambda x, y: (augment(x), y))

for images, labels_batch in train_ds.take(1):
    plt.figure(figsize=(10, 5))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("AI" if labels_batch[i] == 1 else "Real")
        plt.axis("off")
    plt.savefig("Artifacts/augmented_batch.png")
    plt.close()

print("Pipeline ready. Saved Artifacts/augmented_batch.png")