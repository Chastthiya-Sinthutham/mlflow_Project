import mlflow
import pandas as pd
import numpy as np

model_path = r".\mlruns\374442200248369989\models\m-fb2df091a4204bc484286466bb2bb4c7\artifacts"
model = mlflow.pyfunc.load_model(model_path)

# ลองรูปแบบต่างๆ
test_text = "You are stupid and ugly"

print("Testing different input formats:\n")

# Format 1: Single column DataFrame
try:
    input_df = pd.DataFrame([{"tweet_text": test_text}])
    prediction = model.predict(input_df)[0]
    print(f"Format 1 (dict): {prediction}")
except Exception as e:
    print(f"Format 1 failed: {e}")

# Format 2: List of strings
try:
    prediction = model.predict([test_text])[0]
    print(f"Format 2 (list): {prediction}")
except Exception as e:
    print(f"Format 2 failed: {e}")

# Format 3: DataFrame with different column name
try:
    input_df = pd.DataFrame({"text": [test_text]})
    prediction = model.predict(input_df)[0]
    print(f"Format 3 (text column): {prediction}")
except Exception as e:
    print(f"Format 3 failed: {e}")

# Format 4: Series
try:
    input_series = pd.Series([test_text])
    prediction = model.predict(input_series)[0]
    print(f"Format 4 (Series): {prediction}")
except Exception as e:
    print(f"Format 4 failed: {e}")

# Format 5: NumPy array
try:
    input_array = np.array([test_text])
    prediction = model.predict(input_array)[0]
    print(f"Format 5 (NumPy): {prediction}")
except Exception as e:
    print(f"Format 5 failed: {e}")