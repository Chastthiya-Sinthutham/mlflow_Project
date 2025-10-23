import mlflow
import joblib

model_path = r".\mlruns\374442200248369989\models\m-fb2df091a4204bc484286466bb2bb4c7\artifacts"

# โหลด model pickle โดยตรง
model_pkl_path = f"{model_path}/model.pkl"
model = joblib.load(model_pkl_path)

print("Model Structure:")
print(f"Type: {type(model)}")
print(f"Attributes: {dir(model)}")

# ถ้าเป็น Pipeline
if hasattr(model, 'steps'):
    print("\nPipeline Steps:")
    for name, step in model.steps:
        print(f"  - {name}: {type(step)}")

# ตรวจสอบ classes
if hasattr(model, 'classes_'):
    print(f"\nClasses: {model.classes_}")