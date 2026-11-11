import os
import subprocess
import mlflow
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv
from data_pipeline import train_ds, val_ds, test_ds

load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Training")

# 1. Get Git commit hash
try:
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:8]
except Exception:
    git_commit = "unknown"

# 2. Get DVC data version from data.dvc
dvc_version = "unknown"
try:
    with open("data.dvc", "r") as f:
        for line in f:
            if "md5:" in line:
                dvc_version = line.split(":")[1].strip().split(".")[0][:8]
                break
except Exception:
    pass

image_size = 224
seed = 42
learning_rate = 0.001
epochs = 5  # Keep it short for the MLflow demo
batch_size = 32

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(image_size, image_size, 3)),
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

# ==========================================
# RUN 1: MANUAL LOGGING
# ==========================================
class ManualLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch)
        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)

keras.utils.set_random_seed(seed)
model_manual = build_model()

with mlflow.start_run(run_name="baseline_manual"):
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("seed", seed)
    
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("dvc_version", dvc_version)
    mlflow.set_tag("approach", "manual")
    
    logger = ManualLogger()
    model_manual.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[logger], verbose=1)
    
    test_loss, test_acc, test_auc = model_manual.evaluate(test_ds, verbose=0)
    mlflow.log_metric("final_test_loss", test_loss)
    mlflow.log_metric("final_test_accuracy", test_acc)
    mlflow.log_metric("final_test_auc", test_auc)

# ==========================================
# RUN 2: AUTOLOGGING
# ==========================================
keras.utils.set_random_seed(seed)
model_auto = build_model()

# Enable autologging (log_models=False keeps it fast by skipping heavy model uploads)
mlflow.tensorflow.autolog(log_models=False) 

with mlflow.start_run(run_name="baseline_autolog"):
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("dvc_version", dvc_version)
    mlflow.set_tag("approach", "autolog")
    
    model_auto.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    
    test_loss, test_acc, test_auc = model_auto.evaluate(test_ds, verbose=0)
    mlflow.log_metric("final_test_loss", test_loss)
    mlflow.log_metric("final_test_accuracy", test_acc)
    mlflow.log_metric("final_test_auc", test_auc)

mlflow.tensorflow.autolog(disable=True)
print("Done! Check DagsHub Experiments tab.")