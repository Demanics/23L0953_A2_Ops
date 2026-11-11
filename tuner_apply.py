import pandas as pd
import keras_tuner as kt
from tensorflow import keras
from data_pipeline import train_ds, val_ds

image_size = 224
keras.utils.set_random_seed(42)

def build_model(hp):
    dropout_rate = hp.Float("dropout", 0.2, 0.6, step=0.1)
    learning_rate = hp.Choice("learning_rate", [0.001, 0.0003, 0.0001])
    unfrozen_layers = hp.Choice("unfrozen_layers", [0, 20, 40])

    base_model = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
    base_model.trainable = False

    if unfrozen_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-unfrozen_layers:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model

tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=8, directory="Artifacts/tuner_apply", project_name="art", overwrite=True)
tuner.search(train_ds, validation_data=val_ds, epochs=10, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)], verbose=2)

best = tuner.get_best_hyperparameters(1)[0]
print("best learning rate", best.get("learning_rate"))
print("best dropout", best.get("dropout"))
print("best unfrozen layers", best.get("unfrozen_layers"))

rows = []
for trial in tuner.oracle.trials.values():
    rows.append({
        "trial": trial.trial_id, "val_accuracy": trial.score,
        "learning_rate": trial.hyperparameters.get("learning_rate"),
        "dropout": trial.hyperparameters.get("dropout"),
        "unfrozen_layers": trial.hyperparameters.get("unfrozen_layers")
    })

table = pd.DataFrame(rows)
table.to_csv("Artifacts/tuner_trials.csv", index=False)
print(table)