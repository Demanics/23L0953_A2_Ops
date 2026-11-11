import os
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from tensorflow import keras

load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Registry")

# Version 1: the baseline CNN
baseline_model = keras.models.load_model("Artifacts/baseline.keras")
with mlflow.start_run(run_name="register_v1_baseline"):
    mlflow.keras.log_model(baseline_model, name="visionguard", registered_model_name="VisionGuard")
print("Registered version 1 (baseline CNN)")

# Version 2: the transfer-learning model (the better one)
transfer_model = keras.models.load_model("Artifacts/transfer.keras")
with mlflow.start_run(run_name="register_v2_transfer"):
    mlflow.keras.log_model(transfer_model, name="visionguard", registered_model_name="VisionGuard")
print("Registered version 2 (transfer model)")

# Give version 2 the alias "champion"
client = MlflowClient()
client.set_registered_model_alias(name="VisionGuard", alias="champion", version=2)
print("Set alias 'champion' -> version 2")

print("Done! Open the Models/Registry tab on DagsHub.")