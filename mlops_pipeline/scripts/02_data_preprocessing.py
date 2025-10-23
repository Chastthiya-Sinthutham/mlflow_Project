from ast import main
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def preprocess_data(data_path, test_size=0.25, random_state=42):
    """
    Loads raw tweet data, splits it into training and testing sets,
    and logs the resulting datasets as artifacts in MLflow.
    """
    # ตั้งค่า tracking URI ในโค้ดโดยตรง - ใช้ absolute path
    if os.getenv('GITHUB_ACTIONS'):
        # ล้างค่า environment variable ทั้งหมดที่เกี่ยวข้อง
        for key in ['MLFLOW_TRACKING_URI', 'MLFLOW_ARTIFACT_URI', 'MLFLOW_REGISTRY_URI']:
            os.environ.pop(key, None)
        
        # ตั้งค่า tracking URI
        tracking_uri = f"file://{os.getcwd()}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Set MLflow tracking URI to: {tracking_uri}")
    
    mlflow.set_experiment("Cyberbullying Classification - Data Preprocessing v2")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # 1. Load data from CSV
        try:
            df = pd.read_csv(data_path)
            print(f"✓ Data loaded successfully from {data_path}")
        except FileNotFoundError:
            print(f"✗ Error: The file was not found at {data_path}")
            return None
            
        # Drop rows with missing values for simplicity
        df.dropna(subset=['tweet_text', 'cyberbullying_type'], inplace=True)

        # 2. Split the data into training and testing sets
        X = df['tweet_text']
        y = df['cyberbullying_type']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 3. Create a temporary directory to save processed data
        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)

        # Recombine features and target for easy saving
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)
        print(f"Saved processed data to '{processed_data_dir}' directory.")

        # 4. Log parameters and metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))

        # 5. Log the processed data directory as an artifact
        try:
            mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
            print("Logged processed data as artifacts in MLflow.")
        except Exception as e:
            print(f"Error logging artifacts: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Artifact URI: {mlflow.get_artifact_uri()}")
            return None
        
        print("-" * 50)
        print("Data preprocessing run finished.")
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)
        
        # ส่ง run_id กลับไปให้ GitHub Actions
        if os.getenv('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"run_id={run_id}\n")
        
        return run_id

if __name__ == "__main__":
    if os.getenv('GITHUB_ACTIONS'):
        csv_path = "cyberbullying_tweets.csv"
    else:
        csv_path = r"C:\Users\Advice IT\mlflow_Project\cyberbullying_tweets.csv"
    
    print(f"Looking for CSV at: {csv_path}")
    run_id = preprocess_data(data_path=csv_path)
    if run_id:
        print(f"✓ Successfully completed preprocessing with run_id: {run_id}")
    else:
        print("✗ Preprocessing failed")
        exit(1)
