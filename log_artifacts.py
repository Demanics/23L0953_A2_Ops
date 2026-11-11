import os
import sys
import subprocess
import pandas as pd
import mlflow
from dotenv import load_dotenv
from data_pipeline import train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Training")

os.makedirs("Artifacts", exist_ok=True)

# 1. Save the split lists as CSV files
pd.DataFrame({"path": train_paths, "label": train_labels}).to_csv("Artifacts/train_split.csv", index=False)
pd.DataFrame({"path": val_paths, "label": val_labels}).to_csv("Artifacts/validation_split.csv", index=False)
pd.DataFrame({"path": test_paths, "label": test_labels}).to_csv("Artifacts/test_split.csv", index=False)

# 2. Save the package list
with open("Artifacts/requirements.txt", "w") as f:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)

# 3. List of artifacts to attach to the run
artifacts_to_log = [
    "Artifacts/cm_baseline.png",
    "Artifacts/roc_curves.png",
    "Artifacts/train_split.csv",
    "Artifacts/validation_split.csv",
    "Artifacts/test_split.csv",
    "Artifacts/requirements.txt",
    "Artifacts/baseline.keras"
]

with mlflow.start_run(run_name="visionguard_artifacts"):
    mlflow.set_tag("purpose", "task_3_4_artifacts")
    mlflow.set_tag("model", "baseline_cnn")

    for artifact_path in artifacts_to_log:
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)
            print("logged:", artifact_path)
        else:
            print("missing (skipped):", artifact_path)

print("Done! Open the run on DagsHub and check the Artifacts tab.")