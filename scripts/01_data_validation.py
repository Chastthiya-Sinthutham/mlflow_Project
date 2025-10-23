import pandas as pd
import mlflow

def validate_data(data_path):
    """
    Loads the cyberbullying tweets dataset, performs basic validation,
    and logs the results to MLflow.
    """
    mlflow.set_experiment("Cyberbullying Classification - Data Validation v2")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data from CSV
        try:
            df = pd.read_csv(data_path)
            print(f"Data loaded successfully from {data_path}.")
        except FileNotFoundError:
            print(f"Error: The file was not found at {data_path}")
            print("Please update the path in the 'if __name__ == \"__main__\":' block.")
            return

        # 2. Perform validation checks
        num_rows, num_cols = df.shape
        # The dataset has 6 classes: religion, age, gender, ethnicity, not_cyberbullying, other_cyberbullying
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
    # --- IMPORTANT ---
    # Update this path to the actual location of your CSV file.
    # Using raw string (r"...") or forward slashes (/) is recommended for Windows paths.
    csv_path = r"C:\Users\ACE\Documents\mlflow_Project\cyberbullying_tweets.csv"
    validate_data(data_path=csv_path)
