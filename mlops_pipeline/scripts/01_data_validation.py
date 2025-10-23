import os
import pandas as pd
import mlflow

def validate_data(data_path):
    """
    Loads the cyberbullying tweets dataset, performs basic validation,
    and logs the results to MLflow.
    """
    # ตั้งค่า tracking URI ในโค้ดโดยตรง - ใช้ absolute path
    if os.getenv('GITHUB_ACTIONS'):
        # ล้างค่า environment variable ทั้งหมดที่เกี่ยวข้อง
        for key in ['MLFLOW_TRACKING_URI', 'MLFLOW_ARTIFACT_URI', 'MLFLOW_REGISTRY_URI']:
            os.environ.pop(key, None)
        
        tracking_uri = f"file://{os.getcwd()}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        os.environ['MLFLOW_ARTIFACT_URI'] = tracking_uri
        print(f"Set MLflow tracking URI to: {tracking_uri}")
    
    mlflow.set_experiment("Cyberbullying Classification - Data Validation v3")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data from CSV
        try:
            df = pd.read_csv(data_path)
            print(f"✓ Data loaded successfully from {data_path}.")
        except FileNotFoundError:
            print(f"✗ Error: The file was not found at {data_path}")
            print("Please update the path in the 'if __name__ == \"__main__\":' block.")
            return

        # 2. Perform validation checks
        num_rows, num_cols = df.shape
        num_classes = df['cyberbullying_type'].nunique()
        missing_values = df.isnull().sum().sum()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Column names: {df.columns.tolist()}")
        print(f"Number of classes: {num_classes}")
        print(f"Class distribution:\n{df['cyberbullying_type'].value_counts()}")
        print(f"Missing values: {missing_values}")

        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)

        # Check if the data passes our defined criteria
        validation_status = "Success"
        if missing_values > 0 or num_classes < 6:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")
        print("Data validation run finished.")


if __name__ == "__main__":
    if os.getenv('GITHUB_ACTIONS'):
        csv_path = "cyberbullying_tweets.csv"
    else:
        csv_path = r"C:\Users\Advice IT\mlflow_Project\cyberbullying_tweets.csv"
    
    print(f"Looking for CSV at: {csv_path}")
    validate_data(data_path=csv_path)
