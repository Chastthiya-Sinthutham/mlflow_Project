import pandas as pd
import mlflow
import sys
import os
from mlflow.artifacts import download_artifacts

# --- อัปเดต: เพิ่ม import ที่จำเป็น ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def train_evaluate_register(preprocessing_run_id, C=1.0):
    """
    โหลดข้อมูล, ฝึกสอนโมเดล LinearSVC พร้อมปรับค่า C, ประเมินผล,
    และลงทะเบียนโมเดลถ้าผ่านเกณฑ์
    """
    ACCURACY_THRESHOLD = 0.80 
    mlflow.set_experiment("Cyberbullying Classification - Model Training v2")

    # --- อัปเดต: ตั้งชื่อ Run ให้สื่อถึงค่า C ที่ใช้ ---
    with mlflow.start_run(run_name=f"linearsvc_C_{C}"):
        print(f"Starting training run with LinearSVC, C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.set_tag("model_type", "LinearSVC") # เพิ่ม Tag บอกประเภทโมเดล
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        # 1. ดาวน์โหลด Artifacts จาก Preprocessing Run
        try:
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )
            train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
            test_df = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            sys.exit(1)

        X_train = train_df['tweet_text']
        y_train = train_df['cyberbullying_type']
        X_test = test_df['tweet_text']
        y_test = test_df['cyberbullying_type']
        
        # 2. สร้าง Scikit-learn Pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            # --- อัปเดต: ใช้ LinearSVC และส่งค่า C เข้าไป ---
            ('model', LinearSVC(C=C, random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train, y_train)

        # 3. ประเมินผลโมเดล
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)

        # 4. Log Parameters, Metrics, และ Model
        # --- อัปเดต: Log ค่า C แทน alpha ---
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(pipeline, "cyberbullying_classifier_pipeline")

        # 5. ตรวจสอบและลงทะเบียนโมเดล
        if acc >= ACCURACY_THRESHOLD:
            print(f"Model accuracy ({acc:.4f}) meets the threshold. Registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/cyberbullying_classifier_pipeline"
            registered_model = mlflow.register_model(model_uri, "cyberbullying-classifier-prod")
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
        else:
            print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")
        
        print("Training run finished.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script.py <run_id> <C_value>")
        sys.exit(1)
    
    preprocessing_run_id = sys.argv[1]
    C_value = float(sys.argv[2])
    
    print(f"Using preprocessing run_id: {preprocessing_run_id}")
    print(f"Using C value: {C_value}")
    train_evaluate_register(preprocessing_run_id=preprocessing_run_id, C=C_value)

