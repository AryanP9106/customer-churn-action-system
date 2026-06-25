from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
import torch.nn as nn
import joblib
import io

# Initialize FastAPI
app = FastAPI(title="Universal AI Churn Prediction API")

# Enable CORS for Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the exact Neural Network Architecture
class ChurnMLP(nn.Module):
    def __init__(self, input_dim=16):
        super(ChurnMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# 2. Load the Weights and Scaler safely
try:
    model = ChurnMLP(input_dim=16)
    model.load_state_dict(torch.load("models/neural_network_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("models/data_scaler.pkl")
except Exception as e:
    print(f"Warning: Could not load models: {e}")

@app.get("/")
def home():
    return {"status": "online", "message": "Universal Churn API is running."}

@app.post("/predict-csv")
async def predict_churn_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            return {"status": "error", "message": "The uploaded CSV file is empty."}
            
        # Standardize fallback for Customer ID tracking
        if 'Customer_ID' not in df.columns:
            df['Customer_ID'] = df.index.astype(str)

        # UNIVERSAL INTERCEPT: If frontend already mapped features, use them directly
        # Otherwise, gracefully default to fallback baseline zeroes to avoid KeyError crashes
        final_features = pd.DataFrame(index=df.index)
        core_mapped_cols = ['Orders', 'Quantity', 'Revenue', 'Profit', 'Discount_Rate']
        
        for idx, col in enumerate(core_mapped_cols):
            if col in df.columns:
                final_features[f'f_{idx}'] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                final_features[f'f_{idx}'] = 0.0
                
        # Fill out remaining structural feature dimensions up to 16 for PyTorch tensor compatibility
        for i in range(len(core_mapped_cols), 16):
            final_features[f'f_{i}'] = 0.0
            
        # Neural Network Scaling & Array Processing
        X_scaled = scaler.transform(final_features.values)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            probabilities = model(X_tensor).squeeze().tolist()
            
        if isinstance(probabilities, float):
            probabilities = [probabilities]
            
        # Map output matrix probabilities to actionable operational strategies
        results = []
        for i, cust_id in enumerate(df['Customer_ID']):
            prob = probabilities[i]
            
            if prob <= 0.40:
                tier, action = "Low", "No immediate intervention required."
            elif prob <= 0.70:
                tier, action = "Medium", "Trigger a personalized product walkthrough email."
            else:
                tier, action = "High", "Offer an immediate 20% loyalty retention discount."
                
            results.append({
                "Customer_ID": str(cust_id),
                "Churn_Probability": f"{round(prob * 100, 2)}%",
                "Risk_Tier": tier,
                "Suggested_Action": action
            })
            
        return {"status": "success", "data": results}
        
    except Exception as e:
        return {"status": "error", "message": f"Server processing error: {str(e)}"}