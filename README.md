# Customer Churn Action System

An end-to-end AI system that predicts customer churn from **any** uploaded dataset (CSV, XLSX, or TSV), estimates revenue at risk, and recommends a retention action per customer. A PyTorch MLP model serves predictions through a FastAPI backend, with a Streamlit dashboard as the frontend.

The system uses fuzzy column-name matching, so it can work with datasets that don't follow a fixed schema — e.g. a column called `"Tenure in Months"`, `"Monthly Charges"`, or `"Total Revenue"` will automatically be mapped to the model's expected inputs.

---

## Project Structure

```
customer-churn-action-system/
│
├── backend/
│   ├── main.py                  # FastAPI app: column mapping, model loading, /predict-csv endpoint
│   └── models/
│       ├── neural_network_weights.pth   # Trained PyTorch MLP checkpoint
│       └── data_scaler.pkl              # Fitted feature scaler (joblib)
│
├── frontend/
│   └── app.py                   # Streamlit dashboard: upload, inference trigger, risk visualization
│
├── notebook/
│   └── ...                      # Model training / experimentation notebooks
│
├── .gitignore
└── README.md
```

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) – REST API serving predictions
- [PyTorch](https://pytorch.org/) – MLP (Multi-Layer Perceptron) classifier for churn probability
- [scikit-learn](https://scikit-learn.org/) + `joblib` – feature scaling (`StandardScaler`/equivalent, persisted as `data_scaler.pkl`)
- [pandas](https://pandas.pydata.org/) – data ingestion, cleaning, and column standardization
- `CORSMiddleware` – cross-origin access for the Streamlit frontend

**Frontend**
- [Streamlit](https://streamlit.io/) – interactive dashboard (file upload, metrics, charts)
- `pandas` + `requests` – local preprocessing and HTTP calls to the backend

**Model**
- A 3-hidden-layer MLP (`128 → 64 → 32 → 1`) with ReLU activations, dropout regularization, and a sigmoid output for binary churn probability.

**Other**
- Jupyter Notebook – model training/experimentation (see `notebook/`)
- Deployed backend (referenced in `app.py`): `https://customer-churn-action-system.onrender.com`

---

## Installation

### Prerequisites
- Python 3.9+
- `pip`
- Git

### 1. Clone the repository

```bash
git clone https://github.com/AryanP9106/customer-churn-action-system.git
cd customer-churn-action-system
```

### 2. Set up the backend

```bash
cd backend
pip install fastapi uvicorn "python-multipart" pandas torch scikit-learn joblib openpyxl
```

> Make sure `models/neural_network_weights.pth` and `models/data_scaler.pkl` are present inside `backend/models/` before starting the server — the API will start in a degraded state (returning a "Model failed to load" error on `/predict-csv`) if they're missing.

Run the API locally:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 3. Set up the frontend

In a separate terminal:

```bash
cd frontend
pip install streamlit pandas requests openpyxl
```

By default, `app.py` points at the deployed backend (`https://customer-churn-action-system.onrender.com/predict-csv`). To use your local backend instead, update the `api_url` variable in `app.py` to `http://localhost:8000/predict-csv`.

Run the dashboard:

```bash
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

---

## API Reference & Usage

Base URL (deployed): `https://customer-churn-action-system.onrender.com`

### `GET /`

Health check endpoint.

**Response**
```json
{
  "status": "online",
  "message": "Universal Churn API is running."
}
```

---

### `POST /predict-csv`

Accepts a customer dataset file, runs it through the churn model, and returns a per-customer prediction with a recommended retention action.

**Request**

| Field  | Type   | Description                                  |
|--------|--------|-----------------------------------------------|
| `file` | `multipart/form-data` | Customer dataset file. Supported extensions: `.csv`, `.xlsx`, `.tsv` |

**cURL example**
```bash
curl -X POST "https://customer-churn-action-system.onrender.com/predict-csv" \
  -F "file=@customers.csv"
```

**Python example**
```python
import requests

with open("customers.csv", "rb") as f:
    response = requests.post(
        "https://customer-churn-action-system.onrender.com/predict-csv",
        files={"file": ("customers.csv", f, "text/csv")}
    )

print(response.json())
```

#### How the input is processed

The backend doesn't require an exact schema. It strips/normalizes column names and matches them against a set of known aliases, so any of the following (and more) will be recognized automatically:

| Model Feature   | Recognized aliases (examples)                                                    |
|------------------|-----------------------------------------------------------------------------------|
| `Customer_ID`    | `Customer ID`, `CustomerID`, `Customer Id`, `customer_id`, `id`                  |
| `Orders`         | `Order Count`, `Tenure in Months`, `Tenure`, `Recency`, `Engagement`              |
| `Quantity`       | `Qty`, `Number of Referrals`, `Referrals`, `Frequency`, `Purchase Frequency`      |
| `Revenue`        | `Total Revenue`, `Monthly Charge(s)`, `Total Charges`, `Sales`, `Amount`          |
| `Profit`         | `Gross Profit`, `Margin` — if missing, **derived as `Revenue × 0.20`**            |
| `Discount_Rate`  | `Discount Rate`, `Discount` — if missing, **defaults to `0.05`**                 |

If `Customer_ID` can't be matched, the row index is used as the ID instead. Any other expected numeric feature that isn't found defaults to `0.0`. Duplicate rows are dropped and missing values are filled with `0` before feature construction.

**Success response — `200 OK`**
```json
{
  "status": "success",
  "data": [
    {
      "Customer_ID": "10234",
      "Churn_Probability": "78.42%",
      "Revenue": 540.0,
      "Revenue_at_Risk": 423.47,
      "Risk_Tier": "High",
      "Suggested_Action": "Offer an immediate 20% loyalty retention discount."
    },
    {
      "Customer_ID": "10235",
      "Churn_Probability": "22.15%",
      "Revenue": 120.0,
      "Revenue_at_Risk": 26.58,
      "Risk_Tier": "Low",
      "Suggested_Action": "No immediate intervention required."
    }
  ]
}
```

**Response fields**

| Field               | Type    | Description                                                            |
|---------------------|---------|--------------------------------------------------------------------------|
| `Customer_ID`       | string  | Identifier resolved from the uploaded file (or row index fallback)     |
| `Churn_Probability` | string  | Model's predicted churn probability, formatted as a percentage          |
| `Revenue`           | float   | Revenue value parsed from the file (`0.0` if not found)                |
| `Revenue_at_Risk`   | float   | `Revenue × Churn_Probability`, rounded to 2 decimals                    |
| `Risk_Tier`         | string  | `"Low"` (≤40%), `"Medium"` (≤70%), or `"High"` (>70%)                  |
| `Suggested_Action`  | string  | Recommended retention action based on the risk tier                    |

**Risk tier logic**

| Churn Probability | Tier   | Suggested Action                                          |
|--------------------|--------|-------------------------------------------------------------|
| 0% – 40%           | Low    | No immediate intervention required                         |
| 40% – 70%          | Medium | Trigger a personalized product walkthrough email           |
| 70% – 100%         | High   | Offer an immediate 20% loyalty retention discount           |

**Error responses** — always returned with HTTP `200`, distinguished by `status: "error"`:

```json
{ "status": "error", "message": "Model failed to load during startup." }
```
```json
{ "status": "error", "message": "Unsupported file format." }
```
```json
{ "status": "error", "message": "Uploaded CSV is empty." }
```
```json
{ "status": "error", "message": "Server processing error: <details>" }
```

> **Note:** Since the API always returns `200`, the frontend (and any client) should check the `status` field in the response body rather than relying on the HTTP status code.

#### How the frontend consumes this API

`frontend/app.py` mirrors the backend's column-standardization logic locally (for preview/display purposes), then:
1. Lets the user upload a CSV/XLSX/TSV file.
2. Displays detected column mappings and a data preview.
3. On **"Run Batch AI Inference"**, sends the file to `/predict-csv`.
4. Merges the returned predictions back with the original data for display.
5. Powers the **"Revenue At Risk"** tab — aggregating `Revenue_at_Risk` by `Risk_Tier`, ranking top at-risk customers, and charting exposure.

---

## License

No license specified yet — add one (e.g. MIT) if you intend this to be reused or contributed to.
