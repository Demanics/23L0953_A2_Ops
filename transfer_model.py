import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from data_pipeline import train_ds, val_ds

image_size = 224
keras.utils.set_random_seed(42)

base_model = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
base_model.trainable = False

inputs = keras.Input(shape=(image_size, image_size, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])

print("stage 1 trainable layers")
for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

history_one = model.fit(train_ds, validation_data=val_ds, epochs=8,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                                   keras.callbacks.ModelCheckpoint("Artifacts/transfer_stage1.keras", monitor="val_loss", save_best_only=True),
                                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)])

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

print("stage 2 trainable layers")
for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

model.compile(optimizer=keras.optimizers.Adam(0.00001), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])

history_two = model.fit(train_ds, validation_data=val_ds, epochs=8,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                                   keras.callbacks.ModelCheckpoint("Artifacts/transfer.keras", monitor="val_loss", save_best_only=True),
                                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)])

pd.DataFrame(history_one.history).to_csv("Artifacts/transfer_stage1_history.csv")
pd.DataFrame(history_two.history).to_csv("Artifacts/transfer_stage2_history.csv")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_one.history["loss"], label="train")
plt.plot(history_one.history["val_loss"], label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("stage 1 frozen"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_one.history["accuracy"], label="train")
plt.plot(history_one.history["val_accuracy"], label="validation")
plt.xlabel("epoch"); plt.title("stage 1 frozen"); plt.legend()
plt.savefig("Artifacts/transfer_stage1_curves.png"); plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_two.history["loss"], label="train")
plt.plot(history_two.history["val_loss"], label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("stage 2 fine tuned"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_two.history["accuracy"], label="train")
plt.plot(history_two.history["val_accuracy"], label="validation")
plt.xlabel("epoch"); plt.title("stage 2 fine tuned"); plt.legend()
plt.savefig("Artifacts/transfer_stage2_curves.png"); plt.close()