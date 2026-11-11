import os
import mlflow
import matplotlib.pyplot as plt
from tensorflow import keras
from dotenv import load_dotenv

# Load the secrets from the .env file
load_dotenv()

os.makedirs("Artifacts", exist_ok=True)

# Point MLflow to DagsHub using the environment variables
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("toy_mlflow_lab")

keras.utils.set_random_seed(42)

(train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
train_images = train_images[:5000] / 255.0
train_labels = train_labels[:5000]


class MLflowLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch)
        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)


def train_one(learning_rate):
    with mlflow.start_run(run_name="dagshub_lr_" + str(learning_rate)):
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("units", 64)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 128)
        mlflow.set_tag("dataset", "fashion_mnist")
        mlflow.set_tag("platform", "dagshub")

        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logger = MLflowLogger()

        history = model.fit(
            train_images,
            train_labels,
            validation_split=0.2,
            epochs=5,
            batch_size=128,
            callbacks=[logger],
            verbose=2
        )

        final_loss, final_accuracy = model.evaluate(train_images, train_labels, verbose=0)
        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("final_accuracy", final_accuracy)

        plt.figure(figsize=(7, 4))
        plt.plot(history.history["loss"], label="train loss")
        plt.plot(history.history["val_loss"], label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss curve for lr=" + str(learning_rate))
        plt.legend()
        plot_path = "Artifacts/dagshub_curves_lr_" + str(learning_rate) + ".png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        model_path = "Artifacts/dagshub_model_lr_" + str(learning_rate) + ".keras"
        model.save(model_path)
        mlflow.log_artifact(model_path)


train_one(0.01)
train_one(0.001)

print("Done! Go to DagsHub -> Experiments tab to see your runs.")