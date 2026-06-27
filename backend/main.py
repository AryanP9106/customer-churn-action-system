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


COLUMN_ALIASES = {
    "Customer_ID": [
        "Customer_ID",
        "Customer ID",
        "CustomerID",
        "Customer Id",
        "customer_id",
        "id",
    ],
    "Orders": [
        "Orders",
        "Order Count",
        "Order_Count",
        "Tenure in Months",
        "Tenure",
        "Recency",
        "Engagement",
    ],
    "Quantity": [
        "Quantity",
        "Qty",
        "Number of Referrals",
        "Referrals",
        "Frequency",
        "Purchase Frequency",
    ],
    "Revenue": [
        "Revenue",
        "Total Revenue",
        "Monthly Charge",
        "Monthly Charges",
        "Total Charges",
        "Sales",
        "Amount",
    ],
    "Profit": [
        "Profit",
        "Gross Profit",
        "Margin",
    ],
    "Discount_Rate": [
        "Discount_Rate",
        "Discount Rate",
        "Discount",
    ],
}


def normalize_column_name(column_name):
    return "".join(char.lower() for char in str(column_name) if char.isalnum())


def find_column(df, aliases):
    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for alias in aliases:
        match = normalized_columns.get(normalize_column_name(alias))
        if match is not None:
            return match

    return None


def build_model_features(df):
    source_columns = {
        target_column: find_column(df, aliases)
        for target_column, aliases in COLUMN_ALIASES.items()
    }

    if "Customer_ID" not in df.columns:
        id_col = source_columns["Customer_ID"]
        df["Customer_ID"] = (
            df[id_col].astype(str)
            if id_col is not None
            else df.index.astype(str)
        )

    final_features = pd.DataFrame(index=df.index)

    core_mapped_cols = [
        "Orders",
        "Quantity",
        "Revenue",
        "Profit",
        "Discount_Rate"
    ]

    for idx, col in enumerate(core_mapped_cols):
        source_col = source_columns[col]

        if source_col is not None:
            final_features[f"f_{idx}"] = pd.to_numeric(
                df[source_col],
                errors="coerce"
            ).fillna(0.0)

        elif col == "Profit":
            final_features[f"f_{idx}"] = final_features["f_2"] * 0.20

        elif col == "Discount_Rate":
            final_features[f"f_{idx}"] = 0.05

        else:
            final_features[f"f_{idx}"] = 0.0

    for i in range(len(core_mapped_cols), 16):
        final_features[f"f_{i}"] = 0.0

    return final_features


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

        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))

        elif filename.endswith(".tsv"):
            df = pd.read_csv(
                io.BytesIO(contents),
                sep="\t"
            )

        else:
            return {
                "status": "error",
                "message": "Unsupported file format."
            }

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(" ", "_")
        )

        df = df.drop_duplicates()
        df = df.fillna(0)

        if df.empty:
            return {
                "status": "error",
                "message": "Uploaded CSV is empty."
            }

        final_features = build_model_features(df)

        X_scaled = scaler.transform(final_features.values)

        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            probabilities = model(X_tensor).squeeze().tolist()

        if isinstance(probabilities, float):
            probabilities = [probabilities]

        results = []

        for i, cust_id in enumerate(df["Customer_ID"]):

            prob = float(probabilities[i])

            revenue = 0.0

            if "Revenue" in df.columns:
                revenue = pd.to_numeric(
                    df.iloc[i]["Revenue"],
                    errors="coerce"
                )

                if pd.isna(revenue):
                    revenue = 0.0

                revenue = float(revenue)

            revenue_at_risk = round(
                revenue * prob,
                2
            )

            if prob <= 0.40:
                tier = "Low"
                action = "No immediate intervention required."

            elif prob <= 0.70:
                tier = "Medium"
                action = (
                    "Trigger a personalized product walkthrough email."
                )

            else:
                tier = "High"
                action = (
                    "Offer an immediate 20% loyalty retention discount."
                )

            results.append(
                {
                    "Customer_ID": str(cust_id),
                    "Churn_Probability": f"{round(prob * 100, 2)}%",
                    "Revenue": revenue,
                    "Revenue_at_Risk": revenue_at_risk,
                    "Risk_Tier": tier,
                    "Suggested_Action": action
                }
            )

            print("\n========== API DEBUG ==========")

            if len(results) > 0:
                print("First prediction:")
                print(results[0])

            print("Response Keys:")
            print(results[0].keys())

            print("===============================\n")

        return {
            "status": "success",
            "data": results
        }
    
        print(results.keys())

    except Exception as e:
        return {
            "status": "error",
            "message": f"Server processing error: {str(e)}"
        }
