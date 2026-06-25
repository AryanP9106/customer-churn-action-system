from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
import torch.nn as nn
import joblib
import io
from pathlib import Path

app = FastAPI(title="Universal AI Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# MODEL ARCHITECTURE
# ==========================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

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


# ==========================
# LOAD MODEL + SCALER
# ==========================
model = None
scaler = None

try:
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = BASE_DIR / "models" / "neural_network_weights.pth"
    SCALER_PATH = BASE_DIR / "models" / "data_scaler.pkl"

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    input_dim = checkpoint["input_dim"]
    hidden_dims = checkpoint["hidden_dims"]
    dropout_rate = checkpoint["dropout_rate"]

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    print("✓ Model loaded successfully")
    print(f"Input Dim: {input_dim}")
    print(f"Hidden Dims: {hidden_dims}")

except Exception as e:
    print(f"Warning: Could not load models: {e}")


# ==========================
# HEALTH CHECK
# ==========================
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Universal Churn API is running."
    }


# ==========================
# PREDICT CSV
# ==========================
@app.post("/predict-csv")
async def predict_churn_from_csv(file: UploadFile = File(...)):

    if model is None or scaler is None:
        return {
            "status": "error",
            "message": "Model failed to load during startup."
        }

    try:
        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            return {
                "status": "error",
                "message": "Uploaded CSV is empty."
            }

        if "Customer_ID" not in df.columns:
            df["Customer_ID"] = df.index.astype(str)

        final_features = pd.DataFrame(index=df.index)

        core_mapped_cols = [
            "Orders",
            "Quantity",
            "Revenue",
            "Profit",
            "Discount_Rate"
        ]

        for idx, col in enumerate(core_mapped_cols):

            if col in df.columns:
                final_features[f"f_{idx}"] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).fillna(0.0)

            else:
                final_features[f"f_{idx}"] = 0.0

        for i in range(len(core_mapped_cols), 16):
            final_features[f"f_{i}"] = 0.0

        X_scaled = scaler.transform(final_features.values)

        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            probabilities = model(X_tensor).squeeze().tolist()

        if isinstance(probabilities, float):
            probabilities = [probabilities]

        results = []

        for i, cust_id in enumerate(df["Customer_ID"]):

            prob = float(probabilities[i])

            if prob <= 0.40:
                tier = "Low"
                action = "No immediate intervention required."

            elif prob <= 0.70:
                tier = "Medium"
                action = "Trigger a personalized product walkthrough email."

            else:
                tier = "High"
                action = "Offer an immediate 20% loyalty retention discount."

            results.append(
                {
                    "Customer_ID": str(cust_id),
                    "Churn_Probability": f"{round(prob * 100, 2)}%",
                    "Risk_Tier": tier,
                    "Suggested_Action": action
                }
            )

        return {
            "status": "success",
            "data": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Server processing error: {str(e)}"
        }