import mlflow
import yaml

model_path = r".\mlruns\374442200248369989\models\m-fb2df091a4204bc484286466bb2bb4c7\artifacts"

# อ่าน MLmodel file
with open(f"{model_path}/MLmodel", 'r') as f:
    mlmodel = yaml.safe_load(f)
    print("Model Metadata:")
    print(yaml.dump(mlmodel, default_flow_style=False))

# ตรวจสอบ requirements
print("\n" + "="*50)
print("Model Requirements:")
with open(f"{model_path}/requirements.txt", 'r') as f:
    print(f.read())