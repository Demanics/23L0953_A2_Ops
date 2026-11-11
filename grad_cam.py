import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image

image_size = 224
keras.utils.set_random_seed(42)

saved_model = keras.models.load_model("Artifacts/transfer.keras")

base_model = keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(image_size, image_size, 3))
base_model.trainable = False

inputs = keras.Input(shape=(image_size, image_size, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.set_weights(saved_model.get_weights())

conv_layer = base_model.get_layer("Conv_1")
base_to_conv = keras.Model(inputs=base_model.inputs, outputs=conv_layer.output)

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
test_paths = paths[val_end:]
test_labels = labels[val_end:]
class_folders = ["RealArt", "AiArtData"]


def load_image(path):
    picture = Image.open(path).convert("RGB")
    picture = picture.resize((image_size, image_size))
    return np.array(picture).astype("float32")


predictions = []
for path in test_paths:
    image = load_image(path)
    probability = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0][0]
    predictions.append(probability)

correct_numbers = []
wrong_numbers = []
for number in range(len(test_paths)):
    predicted = 1 if predictions[number] >= 0.5 else 0
    if predicted == test_labels[number]:
        correct_numbers.append(number)
    else:
        wrong_numbers.append(number)

correct_numbers = correct_numbers[:4]
wrong_numbers = wrong_numbers[:4]
chosen = correct_numbers + wrong_numbers
print("correct shown:", len(correct_numbers), "wrong shown:", len(wrong_numbers))


def grad_cam(path):
    image = load_image(path)
    image_batch = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        processed = keras.applications.mobilenet_v2.preprocess_input(image_batch)
        conv_output = base_to_conv(processed)
        pooled = tf.reduce_mean(conv_output, axis=(1, 2))
        prediction = model.layers[-1](pooled)

    gradients = tape.gradient(prediction, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output * weights[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap[0].numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 0.0000001)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image_size, image_size)).numpy()[:, :, 0]
    return image, heatmap, float(prediction[0][0])


plt.figure(figsize=(16, 8))
for place in range(len(chosen)):
    number = chosen[place]
    image, heatmap, probability = grad_cam(test_paths[number])
    plt.subplot(2, 4, place + 1)
    plt.imshow(image.astype("uint8"))
    plt.imshow(heatmap, alpha=0.5, cmap="jet")
    true_class = class_folders[test_labels[number]]
    predicted_class = class_folders[1 if probability >= 0.5 else 0]
    status = "CORRECT" if place < len(correct_numbers) else "WRONG"
    plt.title(status + " | true: " + true_class + " | pred: " + predicted_class, fontsize=8)
    plt.axis("off")
plt.savefig("Artifacts/grad_cam.png")
plt.close()
print("saved Artifacts/grad_cam.png")