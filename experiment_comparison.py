import os
import subprocess
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Experiments")

keras.utils.set_random_seed(42)
random.seed(42)

try:
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:8]
except Exception:
    git_commit = "unknown"

dvc_version = "unknown"
try:
    with open("data.dvc", "r") as f:
        for line in f:
            if "md5:" in line:
                dvc_version = line.split(":")[1].strip().split(".")[0][:8]
                break
except Exception:
    pass

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

batch_size = 32


def make_datasets(image_size, use_augment):
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [image_size, image_size])
        return img, tf.cast(label, tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(1000, seed=42).map(load_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if use_augment:
        augment = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1)
        ])
        train_ds = train_ds.map(lambda x, y: (augment(x), y))

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(load_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(backbone, image_size, unfrozen_layers, learning_rate):
    if backbone == "MobileNetV2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    else:
        base = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
        preprocess = keras.applications.resnet50.preprocess_input

    base.trainable = False
    if unfrozen_layers > 0:
        base.trainable = True
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-unfrozen_layers:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


class MLflowLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch)
        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)


experiments = [
    {"name": "1_baseline",        "backbone": "MobileNetV2", "lr": 0.0003,  "augment": False, "unfrozen": 0,  "image_size": 160, "changed": "baseline reference"},
    {"name": "2_lower_lr",        "backbone": "MobileNetV2", "lr": 0.00005, "augment": False, "unfrozen": 0,  "image_size": 160, "changed": "learning rate only"},
    {"name": "3_with_augment",    "backbone": "MobileNetV2", "lr": 0.0003,  "augment": True,  "unfrozen": 0,  "image_size": 160, "changed": "augmentation only"},
    {"name": "4_unfreeze_20",     "backbone": "MobileNetV2", "lr": 0.0003,  "augment": False, "unfrozen": 20, "image_size": 160, "changed": "unfrozen layers only"},
    {"name": "5_bigger_image",    "backbone": "MobileNetV2", "lr": 0.0003,  "augment": False, "unfrozen": 0,  "image_size": 224, "changed": "image size only"},
    {"name": "6_resnet50",        "backbone": "ResNet50",    "lr": 0.0003,  "augment": False, "unfrozen": 0,  "image_size": 160, "changed": "backbone only"},
]

results = []

for exp in experiments:
    print("=== Running:", exp["name"], "===")
    with mlflow.start_run(run_name=exp["name"]):
        mlflow.log_param("backbone", exp["backbone"])
        mlflow.log_param("learning_rate", exp["lr"])
        mlflow.log_param("augment", exp["augment"])
        mlflow.log_param("unfrozen_layers", exp["unfrozen"])
        mlflow.log_param("image_size", exp["image_size"])
        mlflow.log_param("epochs", 4)

        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_version", dvc_version)
        mlflow.set_tag("what_changed", exp["changed"])

        train_ds, val_ds, test_ds = make_datasets(exp["image_size"], exp["augment"])
        model = build_model(exp["backbone"], exp["image_size"], exp["unfrozen"], exp["lr"])

        logger = MLflowLogger()
        history = model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=[logger], verbose=2)

        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_auc", test_auc)

        best_val_acc = max(history.history["val_accuracy"])
        results.append({
            "run": exp["name"],
            "changed": exp["changed"],
            "backbone": exp["backbone"],
            "lr": exp["lr"],
            "augment": exp["augment"],
            "unfrozen": exp["unfrozen"],
            "image_size": exp["image_size"],
            "best_val_accuracy": round(best_val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "test_auc": round(test_auc, 4)
        })

results_table = pd.DataFrame(results)
results_table.to_csv("Artifacts/experiment_comparison.csv", index=False)
print(results_table.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.bar(results_table["run"], results_table["test_accuracy"], color="steelblue")
plt.ylabel("test accuracy")
plt.title("Experiment comparison: test accuracy per run")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("Artifacts/experiment_comparison.png", dpi=120)
plt.close()

with mlflow.start_run(run_name="experiment_comparison"):
    mlflow.set_tag("purpose", "task_3_5_comparison")
    mlflow.log_artifact("Artifacts/experiment_comparison.csv")
    mlflow.log_artifact("Artifacts/experiment_comparison.png")

print("Done! Check DagsHub Experiments -> VisionGuard_Experiments")