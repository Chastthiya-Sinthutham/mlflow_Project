import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import time
from datetime import datetime
import json
import os

print("="*80)
print("🧪 DIRECT MODEL EVALUATION ON FULL TEST SET")
print("="*80)

# ========================================================================================
# LOAD MODEL
# ========================================================================================

print("\n📦 Loading model...")
model_path = r".\mlruns\330612562760685645\models\m-e32c2072bfc742c1a9569c66b038fa48\artifacts"
# Convert to absolute path and proper URI format
absolute_model_path = os.path.abspath(model_path)
model_uri = f"file:///{absolute_model_path.replace(os.sep, '/')}"

print(f"Model path: {absolute_model_path}")
print(f"Model URI: {model_uri}")

try:
    # โหลด MLflow model with proper URI
    model = mlflow.pyfunc.load_model(model_uri)
    
    # โหลด sklearn model สำหรับ decision_function
    sklearn_model = joblib.load(os.path.join(absolute_model_path, "model.pkl"))
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Model type: {type(sklearn_model)}")
    print(f"✓ Classes: {sklearn_model.classes_}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# ========================================================================================
# LOAD TEST DATA
# ========================================================================================

print("\n📊 Loading test data...")
test_data_path = r".\mlruns\364772752496341707\62c697a532b74b5fbc5fc06fb0403348\artifacts\processed_data\test.csv"

# Convert to absolute path
absolute_test_path = os.path.abspath(test_data_path)
print(f"Test data path: {absolute_test_path}")

try:
    test_df = pd.read_csv(absolute_test_path)
    print(f"✓ Test data loaded: {test_df.shape}")
    print(f"✓ Columns: {test_df.columns.tolist()}")
    
    print(f"\n📈 Label Distribution:")
    print(test_df['cyberbullying_type'].value_counts())
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def softmax(scores):
    """Convert decision function scores to probabilities using softmax"""
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def calculate_entropy(probabilities):
    """Calculate Shannon entropy of probability distribution"""
    probs = np.array(probabilities)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def detect_anomaly(confidence, entropy, probabilities, text_length):
    """Detect anomalies based on multiple criteria"""
    CONFIDENCE_THRESHOLD = 0.5
    ENTROPY_THRESHOLD = 1.5
    
    is_anomaly = False
    anomaly_score = 0.0
    reasons = []
    
    # Low confidence
    if confidence < CONFIDENCE_THRESHOLD:
        is_anomaly = True
        anomaly_score += (CONFIDENCE_THRESHOLD - confidence)
        reasons.append(f"Low confidence ({confidence:.4f})")
    
    # High entropy
    if entropy > ENTROPY_THRESHOLD:
        is_anomaly = True
        anomaly_score += (entropy - ENTROPY_THRESHOLD)
        reasons.append(f"High entropy ({entropy:.4f})")
    
    # Uniform distribution
    max_prob = max(probabilities)
    min_prob = min(probabilities)
    prob_range = max_prob - min_prob
    if prob_range < 0.2:
        is_anomaly = True
        anomaly_score += (0.2 - prob_range)
        reasons.append(f"Uniform distribution (range={prob_range:.4f})")
    
    # Text length
    if text_length < 10:
        is_anomaly = True
        anomaly_score += 0.5
        reasons.append(f"Very short text ({text_length})")
    elif text_length > 500:
        is_anomaly = True
        anomaly_score += 0.3
        reasons.append(f"Very long text ({text_length})")
    
    # Multiple similar predictions
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.1:
        is_anomaly = True
        anomaly_score += 0.3
        reasons.append(f"Similar top predictions")
    
    return is_anomaly, anomaly_score, reasons

# ========================================================================================
# PREDICTION
# ========================================================================================

print("\n" + "="*80)
print("🚀 RUNNING PREDICTIONS...")
print("="*80)

start_time = time.time()

# Prepare data
texts = test_df['tweet_text'].tolist()
true_labels = test_df['cyberbullying_type'].tolist()

print(f"\n📝 Predicting {len(texts)} samples...")

# Batch prediction
try:
    # Predictions
    predictions = model.predict(texts)
    
    # Decision scores
    decision_scores = sklearn_model.decision_function(texts)
    
    # Probabilities
    probabilities = softmax(decision_scores)
    
    # Confidence scores
    confidences = []
    for i, pred in enumerate(predictions):
        pred_idx = np.where(sklearn_model.classes_ == pred)[0][0]
        confidence = probabilities[i][pred_idx]
        confidences.append(confidence)
    
    # Entropy
    entropies = [calculate_entropy(prob) for prob in probabilities]
    
    # Anomaly detection
    anomalies = []
    anomaly_scores = []
    anomaly_reasons_list = []
    
    for i, (conf, ent, prob, text) in enumerate(zip(confidences, entropies, probabilities, texts)):
        is_anom, anom_score, reasons = detect_anomaly(conf, ent, prob, len(text))
        anomalies.append(is_anom)
        anomaly_scores.append(anom_score)
        anomaly_reasons_list.append(reasons)
        
        # Progress
        if (i + 1) % 1000 == 0:
            print(f"✓ Processed {i+1}/{len(texts)} samples...")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Prediction completed!")
    print(f"⏱️  Total Time: {elapsed_time:.2f} seconds")
    print(f"📊 Processing Rate: {len(texts)/elapsed_time:.2f} samples/sec")
    
except Exception as e:
    print(f"✗ Error during prediction: {e}")
    exit(1)

# ========================================================================================
# EVALUATION RESULTS
# ========================================================================================

print("\n" + "="*80)
print("📈 COMPREHENSIVE EVALUATION RESULTS")
print("="*80)

# Convert predictions to list if numpy array
predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else predictions

# 1. Overall Performance
accuracy = accuracy_score(true_labels, predictions_list)
correct_count = sum([t == p for t, p in zip(true_labels, predictions_list)])
incorrect_count = len(true_labels) - correct_count

print(f"\n🎯 OVERALL PERFORMANCE")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Total Samples: {len(predictions_list)}")
print(f"   Correct Predictions: {correct_count}")
print(f"   Incorrect Predictions: {incorrect_count}")
print(f"   Error Rate: {(1-accuracy)*100:.2f}%")

# 2. Confidence Statistics
avg_confidence = np.mean(confidences)
std_confidence = np.std(confidences)
min_confidence = np.min(confidences)
max_confidence = np.max(confidences)
median_confidence = np.median(confidences)

print(f"\n📊 CONFIDENCE STATISTICS")
print(f"   Average: {avg_confidence:.4f}")
print(f"   Std Dev: {std_confidence:.4f}")
print(f"   Min: {min_confidence:.4f}")
print(f"   Max: {max_confidence:.4f}")
print(f"   Median: {median_confidence:.4f}")
print(f"   25th Percentile: {np.percentile(confidences, 25):.4f}")
print(f"   75th Percentile: {np.percentile(confidences, 75):.4f}")

# 3. Anomaly Detection Results
anomaly_count = sum(anomalies)
anomaly_rate = (anomaly_count / len(anomalies) * 100)
avg_entropy = np.mean(entropies)

print(f"\n⚠️  ANOMALY DETECTION")
print(f"   Anomalies Detected: {anomaly_count}/{len(anomalies)} ({anomaly_rate:.2f}%)")
print(f"   Average Entropy: {avg_entropy:.4f}")
print(f"   Average Anomaly Score: {np.mean([s for s in anomaly_scores if s > 0]):.4f}")

# Accuracy by anomaly status
anomaly_indices = [i for i, a in enumerate(anomalies) if a]
non_anomaly_indices = [i for i, a in enumerate(anomalies) if not a]

if anomaly_indices:
    anomaly_accuracy = accuracy_score(
        [true_labels[i] for i in anomaly_indices],
        [predictions_list[i] for i in anomaly_indices]
    )
    non_anomaly_accuracy = accuracy_score(
        [true_labels[i] for i in non_anomaly_indices],
        [predictions_list[i] for i in non_anomaly_indices]
    )
    
    print(f"\n🔍 ACCURACY BY ANOMALY STATUS")
    print(f"   Anomalies: {anomaly_accuracy:.4f} ({len(anomaly_indices)} samples)")
    print(f"   Non-Anomalies: {non_anomaly_accuracy:.4f} ({len(non_anomaly_indices)} samples)")

# 4. Accuracy by Confidence Level
print(f"\n📊 ACCURACY BY CONFIDENCE LEVEL")
confidence_bins = [
    (0.0, 0.5, "Very Low"),
    (0.5, 0.7, "Low"),
    (0.7, 0.85, "Medium"),
    (0.85, 0.95, "High"),
    (0.95, 1.0, "Very High")
]

for low, high, label in confidence_bins:
    indices = [i for i, c in enumerate(confidences) if low <= c < high]
    if indices:
        bin_accuracy = accuracy_score(
            [true_labels[i] for i in indices],
            [predictions_list[i] for i in indices]
        )
        avg_conf = np.mean([confidences[i] for i in indices])
        print(f"   {label:12s} [{low:.2f}, {high:.2f}): {bin_accuracy:.4f} "
              f"(avg conf: {avg_conf:.4f}, {len(indices):5d} samples)")

# 5. Detailed Classification Report
print(f"\n📋 DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(
    true_labels,
    predictions_list,
    zero_division=0,
    digits=4
))

# 6. Confusion Matrix
print(f"\n🔢 CONFUSION MATRIX")
print("="*80)
labels = sorted(sklearn_model.classes_)
cm = confusion_matrix(true_labels, predictions_list, labels=labels)

# Pretty print confusion matrix
print(f"\nTrue \\ Predicted", end="")
for label in labels:
    print(f"\t{label[:8]}", end="")
print()
print("-" * 100)

for i, true_label in enumerate(labels):
    print(f"{true_label[:15]:<15}", end="")
    for j in range(len(labels)):
        print(f"\t{cm[i][j]}", end="")
    print()

# 7. Per-Class Detailed Statistics
print(f"\n📊 PER-CLASS DETAILED STATISTICS")
print("="*80)

class_stats = []

for label in labels:
    # Indices
    true_indices = [i for i, t in enumerate(true_labels) if t == label]
    pred_indices = [i for i, p in enumerate(predictions_list) if p == label]
    correct_indices = [i for i, (t, p) in enumerate(zip(true_labels, predictions_list)) 
                      if t == label and p == label]
    
    # Counts
    true_count = len(true_indices)
    pred_count = len(pred_indices)
    correct_count = len(correct_indices)
    
    # Metrics
    precision = correct_count / pred_count if pred_count > 0 else 0
    recall = correct_count / true_count if true_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confidence stats
    class_pred_confidences = [confidences[i] for i in pred_indices]
    avg_conf = np.mean(class_pred_confidences) if class_pred_confidences else 0
    
    # Anomaly stats
    class_anomalies = [anomalies[i] for i in true_indices]
    anomaly_rate_class = (sum(class_anomalies) / len(class_anomalies) * 100) if class_anomalies else 0
    
    # Misclassification analysis
    false_positives = pred_count - correct_count
    false_negatives = true_count - correct_count
    
    class_stats.append({
        'label': label,
        'true_count': true_count,
        'pred_count': pred_count,
        'correct': correct_count,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_conf': avg_conf,
        'anomaly_rate': anomaly_rate_class,
        'false_pos': false_positives,
        'false_neg': false_negatives
    })
    
    print(f"\n{label}:")
    print(f"   Support (True Count): {true_count}")
    print(f"   Predicted Count: {pred_count}")
    print(f"   Correct Predictions: {correct_count}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Avg Confidence: {avg_conf:.4f}")
    print(f"   Anomaly Rate: {anomaly_rate_class:.2f}%")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")

# 8. Most Confident Correct Predictions
print(f"\n✅ TOP 10 MOST CONFIDENT CORRECT PREDICTIONS")
print("="*80)

correct_data = [(i, confidences[i], true_labels[i], predictions_list[i], texts[i]) 
                for i in range(len(predictions_list)) 
                if true_labels[i] == predictions_list[i]]
correct_data.sort(key=lambda x: x[1], reverse=True)

for i, (idx, conf, label, pred, text) in enumerate(correct_data[:10], 1):
    print(f"\n{i}. Confidence: {conf:.4f}")
    print(f"   Label: {label}")
    print(f"   Text: {text[:100]}...")

# 9. Most Confident Incorrect Predictions
print(f"\n❌ TOP 10 MOST CONFIDENT INCORRECT PREDICTIONS")
print("="*80)

incorrect_data = [(i, confidences[i], true_labels[i], predictions_list[i], texts[i]) 
                  for i in range(len(predictions_list)) 
                  if true_labels[i] != predictions_list[i]]
incorrect_data.sort(key=lambda x: x[1], reverse=True)

for i, (idx, conf, true, pred, text) in enumerate(incorrect_data[:10], 1):
    print(f"\n{i}. Confidence: {conf:.4f}")
    print(f"   True: {true}")
    print(f"   Predicted: {pred}")
    print(f"   Text: {text[:100]}...")

# 10. Lowest Confidence Predictions
print(f"\n⚠️  TOP 10 LOWEST CONFIDENCE PREDICTIONS (Potential Issues)")
print("="*80)

low_conf_data = [(i, confidences[i], true_labels[i], predictions_list[i], texts[i], anomalies[i]) 
                 for i in range(len(predictions_list))]
low_conf_data.sort(key=lambda x: x[1])

for i, (idx, conf, true, pred, text, is_anom) in enumerate(low_conf_data[:10], 1):
    correct = "✓" if true == pred else "✗"
    print(f"\n{i}. Confidence: {conf:.4f} | Anomaly: {is_anom} | {correct}")
    print(f"   True: {true}")
    print(f"   Predicted: {pred}")
    print(f"   Text: {text[:100]}...")

# 11. Misclassification Analysis
print(f"\n🔍 COMMON MISCLASSIFICATION PATTERNS")
print("="*80)

# Count misclassifications
misclass_pairs = {}
for true, pred in zip(true_labels, predictions_list):
    if true != pred:
        pair = (true, pred)
        misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1

# Sort by frequency
sorted_misclass = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 most common misclassifications:")
for i, ((true, pred), count) in enumerate(sorted_misclass[:10], 1):
    percentage = (count / len(predictions_list)) * 100
    print(f"{i:2d}. {true:20s} → {pred:20s}: {count:4d} times ({percentage:.2f}%)")

# ========================================================================================
# SAVE RESULTS
# ========================================================================================

print(f"\n💾 SAVING RESULTS...")

# 1. Detailed predictions
results_df = pd.DataFrame({
    'tweet_text': texts,
    'true_label': true_labels,
    'predicted_label': predictions_list,
    'confidence': confidences,
    'entropy': entropies,
    'is_anomaly': anomalies,
    'anomaly_score': anomaly_scores,
    'correct': [t == p for t, p in zip(true_labels, predictions_list)],
    'text_length': [len(t) for t in texts]
})

results_df.to_csv('direct_model_test_results.csv', index=False)
print(f"✓ Detailed results saved to: direct_model_test_results.csv")

# 2. Summary statistics
summary = {
    'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_path': model_path,
    'test_data_path': test_data_path,
    'total_samples': len(predictions_list),
    'processing_time_seconds': elapsed_time,
    'samples_per_second': len(predictions_list) / elapsed_time,
    'overall_accuracy': float(accuracy),
    'correct_predictions': int(correct_count),
    'incorrect_predictions': int(incorrect_count),
    'avg_confidence': float(avg_confidence),
    'std_confidence': float(std_confidence),
    'min_confidence': float(min_confidence),
    'max_confidence': float(max_confidence),
    'median_confidence': float(median_confidence),
    'anomaly_count': int(anomaly_count),
    'anomaly_rate': float(anomaly_rate),
    'avg_entropy': float(avg_entropy),
    'class_statistics': class_stats
}

with open('direct_model_test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved to: direct_model_test_summary.json")

# 3. Confusion matrix
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv('direct_model_confusion_matrix.csv')
print(f"✓ Confusion matrix saved to: direct_model_confusion_matrix.csv")

# 4. Class statistics
class_stats_df = pd.DataFrame(class_stats)
class_stats_df.to_csv('direct_model_class_statistics.csv', index=False)
print(f"✓ Class statistics saved to: direct_model_class_statistics.csv")

print("\n" + "="*80)
print("🎉 EVALUATION COMPLETE!")
print("="*80)
print(f"\nFiles generated:")
print(f"  1. direct_model_test_results.csv - All predictions with details")
print(f"  2. direct_model_test_summary.json - Summary statistics")
print(f"  3. direct_model_confusion_matrix.csv - Confusion matrix")
print(f"  4. direct_model_class_statistics.csv - Per-class statistics")
print("="*80)