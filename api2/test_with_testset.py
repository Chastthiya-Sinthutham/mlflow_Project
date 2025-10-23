import pandas as pd
import requests
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# โหลดข้อมูล test set
test_data_path = r".\mlruns\546436060077246528\0784c6c37f594d2eb07420948776bfd2\artifacts\processed_data\test.csv"
test_df = pd.read_csv(test_data_path)

print(f"📊 Test Data Shape: {test_df.shape}")
print(f"📋 Columns: {test_df.columns.tolist()}\n")

# เลือกตัวอย่างข้อมูล
sample_size = 100  # ทดสอบ 100 ตัวอย่างก่อน
test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

print(f"🧪 Testing with {len(test_sample)} samples...\n")

# API endpoint
API_URL = "http://localhost:8000/predict"

# ทดสอบทีละข้อความ
predictions = []
true_labels = []
confidences = []
errors = []

for idx, row in test_sample.iterrows():
    tweet_text = row['tweet_text']
    true_label = row['cyberbullying_type']
    
    try:
        response = requests.post(
            API_URL,
            json={"tweet_text": tweet_text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted_label = result['prediction']
            confidence = result['confidence']
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
            confidences.append(confidence)
            
            # แสดงผลทุก 10 ข้อความ
            if (len(predictions)) % 10 == 0:
                print(f"✓ Processed {len(predictions)}/{len(test_sample)} samples...")
        else:
            errors.append({"text": tweet_text, "error": response.text})
            
    except Exception as e:
        errors.append({"text": tweet_text, "error": str(e)})
    
    time.sleep(0.01)

print(f"\n✅ Completed: {len(predictions)}/{len(test_sample)} predictions")

if errors:
    print(f"❌ Errors: {len(errors)}")

# คำนวณ metrics
print("\n" + "="*60)
print("📈 EVALUATION RESULTS")
print("="*60)

accuracy = accuracy_score(true_labels, predictions)
avg_confidence = sum(confidences) / len(confidences) if confidences else 0

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print(f"📊 Average Confidence: {avg_confidence:.4f}")
print(f"📊 Min Confidence: {min(confidences):.4f}")
print(f"📊 Max Confidence: {max(confidences):.4f}")

print("\n📊 Classification Report:")
print(classification_report(true_labels, predictions, zero_division=0))

print("\n🔢 Confusion Matrix:")
labels = sorted(list(set(true_labels + predictions)))
cm = confusion_matrix(true_labels, predictions, labels=labels)
print("\nLabels:", labels)
print(cm)

# แสดงตัวอย่างที่ทำนายผิด พร้อม confidence
print("\n" + "="*60)
print("❌ INCORRECT PREDICTIONS (First 10) with Confidence")
print("="*60)

incorrect_count = 0
for i, (true, pred, conf, row) in enumerate(zip(true_labels, predictions, confidences, test_sample.iterrows())):
    if true != pred and incorrect_count < 10:
        print(f"\n{incorrect_count + 1}. Text: {row[1]['tweet_text'][:80]}...")
        print(f"   True: {true}")
        print(f"   Predicted: {pred}")
        print(f"   Confidence: {conf:.4f}")
        incorrect_count += 1

# แสดงตัวอย่างที่มี confidence ต่ำ
print("\n" + "="*60)
print("⚠️  LOW CONFIDENCE PREDICTIONS (First 10)")
print("="*60)

low_conf_data = sorted(zip(confidences, predictions, true_labels, test_sample.iterrows()), key=lambda x: x[0])
for i, (conf, pred, true, row) in enumerate(low_conf_data[:10]):
    print(f"\n{i + 1}. Text: {row[1]['tweet_text'][:80]}...")
    print(f"   True: {true}")
    print(f"   Predicted: {pred}")
    print(f"   Confidence: {conf:.4f}")
    print(f"   Correct: {'✓' if true == pred else '✗'}")

# สรุปผลตาม class พร้อม confidence
print("\n" + "="*60)
print("📊 PERFORMANCE BY CLASS with Average Confidence")
print("="*60)

for label in labels:
    true_indices = [i for i, t in enumerate(true_labels) if t == label]
    pred_indices = [i for i, p in enumerate(predictions) if p == label]
    correct_indices = [i for i, (t, p) in enumerate(zip(true_labels, predictions)) if t == label and p == label]
    
    true_count = len(true_indices)
    pred_count = len(pred_indices)
    correct = len(correct_indices)
    
    if true_count > 0:
        precision = correct / pred_count if pred_count > 0 else 0
        recall = correct / true_count
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # คำนวณ average confidence สำหรับ class นี้
        class_confidences = [confidences[i] for i in pred_indices]
        avg_class_conf = sum(class_confidences) / len(class_confidences) if class_confidences else 0
        
        print(f"\n{label}:")
        print(f"  True samples: {true_count}")
        print(f"  Predicted: {pred_count}")
        print(f"  Correct: {correct}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Avg Confidence: {avg_class_conf:.4f}")

# บันทึกผลลัพธ์
results_df = pd.DataFrame({
    'tweet_text': [row[1]['tweet_text'] for row in test_sample.iterrows()],
    'true_label': true_labels,
    'predicted_label': predictions,
    'confidence': confidences,
    'correct': [t == p for t, p in zip(true_labels, predictions)]
})

results_df.to_csv('api_test_results.csv', index=False)
print(f"\n💾 Results saved to: api_test_results.csv")