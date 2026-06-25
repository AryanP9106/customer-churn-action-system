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

# 2. Load the Weights and Scaler
try:
    model = ChurnMLP(input_dim=16)
    model.load_state_dict(torch.load("models/neural_network_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("models/data_scaler.pkl")
except Exception as e:
    print(f"Warning: Could not load models. Make sure they exist in the models/ folder. Error: {e}")

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
            
        if 'Customer_ID' not in df.columns:
            df['Customer_ID'] = df.index.astype(str)

        # FOOLPROOF AGGREGATION: Dynamic fallback dictionary
        agg_dict = {}
        for col in ['Orders', 'Quantity', 'Revenue', 'Profit', 'Discount_Rate']:
            if col in df.columns:
                agg_dict[col] = 'mean' if 'Rate' in col else 'sum'
            else:
                df[col] = 0.0
                agg_dict[col] = 'sum'
                
        if 'Order_ID' in df.columns:
            agg_dict['Order_ID'] = 'count'
        elif 'Orders' in df.columns:
            agg_dict['Orders'] = 'count'

        # Safely aggregate customer profiles
        cust_df = df.groupby('Customer_ID').agg(agg_dict).reset_index()
        
        # Ensure matrix is exactly 16 dimensions
        final_features = pd.DataFrame(index=cust_df.index)
        feature_cols = ['Orders', 'Quantity', 'Revenue', 'Profit', 'Discount_Rate']
        
        for idx, col in enumerate(feature_cols):
            if col in cust_df.columns:
                final_features[f'f_{idx}'] = cust_df[col]
            else:
                final_features[f'f_{idx}'] = 0.0
                
        for i in range(len(feature_cols), 16):
            final_features[f'f_{i}'] = 0.0
            
        # Neural Network Inference
        X_scaled = scaler.transform(final_features.values)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            probabilities = model(X_tensor).squeeze().tolist()
            
        if isinstance(probabilities, float):
            probabilities = [probabilities]
            
        # Business Strategy Mapping
        results = []
        for i, cust_id in enumerate(cust_df['Customer_ID']):
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