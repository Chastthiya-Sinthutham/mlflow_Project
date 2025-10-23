import mlflow
import yaml
import os

model_path = r".\mlruns\374442200248369989\models\m-fb2df091a4204bc484286466bb2bb4c7\artifacts"

print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

# 1. ดู MLmodel metadata
print("\n1. MLmodel Metadata:")
with open(f"{model_path}/MLmodel", 'r') as f:
    mlmodel = yaml.safe_load(f)
    print(yaml.dump(mlmodel, default_flow_style=False))

# 2. ดู Model Accuracy
run_id = "a60b4347a28443fca0b87859cdc5cc0f"
metrics_path = f"mlruns/374442200248369989/{run_id}/metrics/accuracy"
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        print(f"\n2. Model Accuracy: {f.read().strip()}")

# 3. ดู Parameters
print("\n3. Model Parameters:")
params_dir = f"mlruns/374442200248369989/{run_id}/params"
for param_file in os.listdir(params_dir):
    with open(f"{params_dir}/{param_file}", 'r') as f:
        print(f"   {param_file}: {f.read().strip()}")

# 4. ลอง inspect model
print("\n4. Model Type:")
model = mlflow.pyfunc.load_model(model_path)
print(f"   Model class: {type(model)}")
print(f"   Model flavor: {model.metadata.flavors}")

# 5. ดู preprocessing run
preprocessing_run_id = None
params_dir = f"mlruns/374442200248369989/{run_id}/params"
if os.path.exists(f"{params_dir}/preprocessing_run_id"):
    with open(f"{params_dir}/preprocessing_run_id", 'r') as f:
        preprocessing_run_id = f.read().strip()
        print(f"\n5. Preprocessing Run ID: {preprocessing_run_id}")