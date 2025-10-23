import pandas as pd
import mlflow
import sys

def load_and_predict():
    """
    จำลองสถานการณ์ใช้งานจริง โดยการโหลดโมเดลที่ลงทะเบียนไว้
    มาใช้ทำนายผลกับข้อมูลใหม่ และเปรียบเทียบกับคำตอบที่ถูกต้อง
    """
    MODEL_NAME = "cyberbullying-classifier-prod"
    MODEL_STAGE = "Staging" # เปลี่ยนเป็น "Production" หลังจากเลื่อนขั้นโมเดล

    print(f"กำลังโหลดโมเดล '{MODEL_NAME}' จาก stage '{MODEL_STAGE}'...")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nเกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        print(f"กรุณาตรวจสอบว่ามีโมเดลเวอร์ชันสำหรับ '{MODEL_NAME}' อยู่ใน stage '{MODEL_STAGE}' ใน MLflow UI แล้ว")
        sys.exit(1)

    # เตรียมข้อมูลตัวอย่างใหม่สำหรับทำนายผล
    sample_tweets = [
        "This is just a regular tweet, nothing to see here.",
        "Christians are all the same, they should go back to their country",
        "You are so old and slow, grandpa.",
        "Go back to Africa you monkey",
        "I love hanging out with my friends, #goodvibes"
    ]
    
    # --- ใหม่: เพิ่มลิสต์ของคำตอบที่ถูกต้อง (เฉลย) ---
    # ลำดับต้องตรงกับ sample_tweets
    actual_labels = [
        "not_cyberbullying",
        "religion",
        "age",
        "ethnicity",
        "not_cyberbullying"
    ]

    sample_df = pd.DataFrame(sample_tweets, columns=['tweet_text'])
    
    # --- แก้ไข: ส่งข้อมูลเฉพาะคอลัมน์ 'tweet_text' (Series) เข้าไปทำนาย ---
    predictions = model.predict(sample_df['tweet_text'])

    # แสดงผลลัพธ์พร้อมการเปรียบเทียบ
    print("-" * 50)
    correct_predictions = 0
    # --- อัปเดต: วนลูปแสดงผลพร้อมเฉลย ---
    for tweet, prediction, actual in zip(sample_tweets, predictions, actual_labels):
        is_correct = (prediction == actual)
        if is_correct:
            correct_predictions += 1
            
        print(f"Tweet         : \"{tweet}\"")
        print(f"--> ทำนายเป็น    : {prediction}")
        print(f"--> เฉลย        : {actual}")
        print(f"--> ผลลัพธ์      : {'ถูกต้อง' if is_correct else 'ผิด'}\n")

    print("-" * 50)
    print(f"สรุป: ทำนายถูกต้อง {correct_predictions} จาก {len(sample_tweets)} ตัวอย่าง")
    print("-" * 50)


if __name__ == "__main__":
    load_and_predict()

