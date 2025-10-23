import pandas as pd
import os

preprocessing_run_id = "0784c6c37f594d2eb07420948776bfd2"
train_data_path = f"mlruns/546436060077246528/{preprocessing_run_id}/artifacts/processed_data/train.csv"
test_data_path = f"mlruns/546436060077246528/{preprocessing_run_id}/artifacts/processed_data/test.csv"

print("Checking preprocessing artifacts...\n")

if os.path.exists(train_data_path):
    train_df = pd.read_csv(train_data_path)
    print("Training Data Sample:")
    print(train_df.head())
    print(f"\nColumns: {train_df.columns.tolist()}")
    print(f"Shape: {train_df.shape}")
    print(f"\nLabel distribution:")
    print(train_df['cyberbullying_type'].value_counts() if 'cyberbullying_type' in train_df.columns else train_df.iloc[:, -1].value_counts())
else:
    print(f"✗ Training data not found at: {train_data_path}")

if os.path.exists(test_data_path):
    test_df = pd.read_csv(test_data_path)
    print(f"\n\nTest Data Shape: {test_df.shape}")
else:
    print(f"✗ Test data not found at: {test_data_path}")