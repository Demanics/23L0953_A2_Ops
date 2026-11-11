import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from data_pipeline import train_ds, val_ds, test_ds

image_size = 224
keras.utils.set_random_seed(42)

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

model.summary()

for layer in model.layers:
    print(layer.name, layer.output.shape, layer.count_params())

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("Artifacts/baseline.keras", monitor="val_loss", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

pd.DataFrame(history.history).to_csv("Artifacts/baseline_history.csv")

best_epoch = history.history["val_loss"].index(min(history.history["val_loss"])) + 1
print("best epoch", best_epoch)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.axvline(best_epoch - 1, color="red", linestyle="--")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.axvline(best_epoch - 1, color="red", linestyle="--")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("Artifacts/baseline_curves.png")
plt.close()