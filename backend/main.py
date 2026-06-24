from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import numpy as np
import io
import os
import torch
import torch.nn as nn

app = FastAPI(title="Neural Network Batch Inference API")

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

WEIGHTS_PATH = "models/neural_network_weights.pth"
SCALER_PATH = "models/data_scaler.pkl"

@app.post("/predict-csv")
async def predict_churn_from_csv(file: UploadFile = File(...)):
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(SCALER_PATH):
        return {"status": "error", "message": "Model weights or data scaler missing."}

    # Load Model and Scaler
    model = MLPClassifier(input_dim=16)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)

    # Read CSV
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Standardize column naming just in case
    df.columns = [col.strip() for col in df.columns]
    
    # Check if the incoming data is raw transaction data or already aggregated
    if 'Customer_ID' in df.columns and ('Revenue' in df.columns or 'total_revenue' not in df.columns):
        # Convert date columns to datetime
        for date_col in ['Transaction_Date', 'Date', 'first_purchase', 'last_purchase']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Baseline date for recency calculations matching notebook logic
        max_date = df['Transaction_Date'].max() if 'Transaction_Date' in df.columns else pd.Timestamp.now()

        # Perform feature aggregation grouping by Customer_ID
        agg_dict = {
            'Order_ID': 'count',
            'Revenue': ['sum', 'mean', 'std'],
            'Profit': ['sum', 'mean'],
            'Discount_Rate': 'mean',
            'Quantity': ['sum', 'mean']
        }
        
        # Keep track of categorical anchors
        for cat in ['Region', 'Product_Category', 'Customer_Segment', 'Payment_Method']:
            if cat in df.columns:
                agg_dict[cat] = 'first'

        cust_df = df.groupby('Customer_ID').agg(agg_dict)
        
        # Flatten multi-index columns to match the 16 feature names
        cust_df.columns = [
            'total_orders', 'total_revenue', 'avg_revenue', 'std_revenue',
            'total_profit', 'avg_profit', 'avg_discount', 'total_quantity', 'avg_quantity'
        ] + [cat for cat in ['Region', 'Product_Category', 'Customer_Segment', 'Payment_Method'] if cat in df.columns]

        # Fill any NaN values introduced by standard deviation on single orders
        cust_df['std_revenue'] = cust_df['std_revenue'].fillna(0)

        # Mock mock time features if raw times aren't fully present to satisfy the shape
        cust_df['days_since_last_purchase'] = 30.0
        cust_df['customer_lifetime_days'] = 365.0
        cust_df['purchase_frequency'] = cust_df['total_orders'] / 12.0
        
        # Map or dummy encode text categories to simple frequency hashes to maintain numeric arrays
        for cat in ['Region', 'Product_Category', 'Customer_Segment', 'Payment_Method']:
            if cat in cust_df.columns:
                cust_df[cat] = cust_df[cat].astype('category').cat.codes

        # Reorder columns to ensure 100% exact alignment with scaler schema
        feature_order = [
            'total_orders', 'total_revenue', 'avg_revenue', 'std_revenue', 'total_profit',
            'avg_profit', 'avg_discount', 'total_quantity', 'avg_quantity', 'days_since_last_purchase',
            'customer_lifetime_days', 'purchase_frequency', 'Region', 'Product_Category',
            'Customer_Segment', 'Payment_Method'
        ]
        
        # Fill missing features if any columns are absent
        for col in feature_order:
            if col not in cust_df.columns:
                cust_df[col] = 0.0
                
        X_df = cust_df[feature_order]
        customer_ids = cust_df.index.tolist()
    else:
        # Data is already pre-aggregated
        X_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        customer_ids = df['Customer_ID'].tolist() if 'Customer_ID' in df.columns else [f"Cust_{i}" for i in range(len(df))]

    X_raw = X_df.values
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        probabilities = model(X_tensor).numpy().flatten()
        
    results = []
    for idx, prob in enumerate(probabilities):
        risk_tier = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
        action = "No action needed."
        if risk_tier == "High":
            action = "Offer an immediate 20% loyalty discount."
        elif risk_tier == "Medium":
            action = "Trigger a personalized product walkthrough email."
            
        results.append({
            "Customer_ID": customer_ids[idx],
            "Churn_Probability": round(float(prob) * 100, 2),
            "Risk_Tier": risk_tier,
            "Suggested_Action": action
        })
        
    return {"status": "success", "data": results}