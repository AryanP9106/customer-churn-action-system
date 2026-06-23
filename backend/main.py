from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import io
import os

app = FastAPI(title="Batch Churn Prediction API")

MODEL_PATH = "models/xgboost_model.pkl"

@app.post("/predict-csv")
async def predict_churn_from_csv(file: UploadFile = File(...)):
    # Safeguard if the model file isn't generated yet
    if not os.path.exists(MODEL_PATH):
        return {"status": "error", "message": "Model file not found. Please train the model first."}
        
    model = joblib.load(MODEL_PATH)
    
    # Read the uploaded CSV bytes
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Clean/Preprocess incoming data to match the training schema
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Run bulk predictions
    probabilities = model.predict_proba(df)[:, 1]
    
    results = []
    for idx, prob in enumerate(probabilities):
        risk_tier = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
        
        action = "No action needed."
        if risk_tier == "High":
            action = "Offer an immediate 20% loyalty discount."
        elif risk_tier == "Medium":
            action = "Trigger a personalized feature walkthrough email."
            
        results.append({
            "Customer_Row": idx + 1,
            "Churn_Probability": round(float(prob) * 100, 2),
            "Risk_Tier": risk_tier,
            "Suggested_Action": action
        })
        
    return {"status": "success", "data": results}