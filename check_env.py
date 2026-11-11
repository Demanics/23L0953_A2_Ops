import sys
import platform
import tensorflow as tf
import PIL
import pandas as pd
import matplotlib

print("Python:", sys.version.split()[0])
print("OS:", platform.system(), platform.release())
print("TensorFlow:", tf.__version__)
print("Pillow:", PIL.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs found:", gpus)

if len(gpus) > 0:
    print("GPU is available -> training will use GPU")
else:
    print("No GPU -> TensorFlow will run on CPU")