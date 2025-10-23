from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
import os
import joblib

app = FastAPI(title="Cyberbullying Classifier API", version="1.0")

# Label mapping (ตาม classes ของ model)
LABEL_MAP = {
    "age": "age",
    "ethnicity": "ethnicity", 
    "gender": "gender",
    "not_cyberbullying": "not_cyberbullying",
    "other_cyberbullying": "other_cyberbullying",
    "religion": "religion"
}

# Global variables
model = None
sklearn_model = None

# โหลดโมเดลตอน startup
@app.on_event("startup")
async def load_model():
    global model, sklearn_model
    try:
        # โหลด model จาก local path โดยตรง
        model_path = r".\mlruns\330612562760685645\models\m-e32c2072bfc742c1a9569c66b038fa48\artifacts"
        model = mlflow.pyfunc.load_model(model_path)
        
        # โหลด sklearn model โดยตรงเพื่อเข้าถึง decision_function
        sklearn_model = joblib.load(f"{model_path}/model.pkl")
        
        print(f"✓ Model loaded from local path: {model_path}")
        print(f"✓ Classes: {sklearn_model.classes_}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
        sklearn_model = None

class PredictionRequest(BaseModel):
    tweet_text: str

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    prediction: str
    confidence: float
    all_scores: dict
    model_version: str
    text_length: int

class BatchPredictionResponse(BaseModel):
    predictions: list[str]
    confidences: list[float]
    all_scores: list[dict]
    count: int

@app.get("/")
async def root():
    return {
        "message": "Cyberbullying Classifier API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0",
        "supported_labels": list(LABEL_MAP.keys())
    }

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

def softmax(scores):
    """Convert decision function scores to probabilities using softmax"""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or sklearn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # ใช้ list format
        prediction = model.predict([request.tweet_text])[0]
        
        # ดึง decision function scores
        decision_scores = sklearn_model.decision_function([request.tweet_text])[0]
        
        # แปลง scores เป็น probabilities ด้วย softmax
        probabilities = softmax(decision_scores)
        
        # หา confidence (probability ของ class ที่ทำนาย)
        predicted_class_idx = np.argmax(decision_scores)
        confidence = float(probabilities[predicted_class_idx])
        
        # สร้าง dict ของ scores ทุก class
        all_scores = {
            class_name: {
                "probability": float(probabilities[i]),
                "decision_score": float(decision_scores[i])
            }
            for i, class_name in enumerate(sklearn_model.classes_)
        }
        
        # Debug
        print(f"Debug - Input: {request.tweet_text}")
        print(f"Debug - Prediction: {prediction}")
        print(f"Debug - Confidence: {confidence:.4f}")
        
        return PredictionResponse(
            prediction=str(prediction),
            confidence=confidence,
            all_scores=all_scores,
            model_version="1",
            text_length=len(request.tweet_text)
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(texts: list[str]):
    if model is None or sklearn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # ใช้ list format
        predictions = model.predict(texts)
        
        # ดึง decision function scores
        decision_scores = sklearn_model.decision_function(texts)
        
        # แปลงเป็น probabilities
        probabilities = np.array([softmax(scores) for scores in decision_scores])
        
        # หา confidence ของแต่ละ prediction
        confidences = []
        all_scores_list = []
        
        for i, pred in enumerate(predictions):
            pred_idx = np.where(sklearn_model.classes_ == pred)[0][0]
            confidence = float(probabilities[i][pred_idx])
            confidences.append(confidence)
            
            # สร้าง scores dict สำหรับแต่ละตัวอย่าง
            scores_dict = {
                class_name: {
                    "probability": float(probabilities[i][j]),
                    "decision_score": float(decision_scores[i][j])
                }
                for j, class_name in enumerate(sklearn_model.classes_)
            }
            all_scores_list.append(scores_dict)
        
        return BatchPredictionResponse(
            predictions=[str(p) for p in predictions],
            confidences=confidences,
            all_scores=all_scores_list,
            count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "Cyberbullying Classifier",
        "model_version": "1",
        "labels": list(LABEL_MAP.keys()),
        "description": "Text classification model for detecting cyberbullying in tweets",
        "model_type": "LinearSVC with TF-IDF",
        "classes": sklearn_model.classes_.tolist() if sklearn_model else [],
        "features": "Confidence scores calculated using softmax on decision function"
    }