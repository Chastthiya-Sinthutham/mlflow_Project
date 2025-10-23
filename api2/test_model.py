import mlflow
import pandas as pd

# โหลด model
model_path = r".\mlruns\374442200248369989\models\m-fb2df091a4204bc484286466bb2bb4c7\artifacts"
model = mlflow.pyfunc.load_model(model_path)

# ทดสอบด้วยข้อความหลายแบบ
test_texts = [
    "You are stupid and ugly",
    "I hate you so much",
    "Have a nice day",
    "You are beautiful",
    "Go kill yourself",
    "they gonna bully this nigga at school",
    "I hope you die",
    "Great work!"
]

print("Testing model predictions:\n")
for text in test_texts:
    input_data = pd.DataFrame([{"tweet_text": text}])
    prediction = model.predict(input_data)[0]
    print(f"Text: {text[:50]}")
    print(f"Prediction: {prediction}\n")