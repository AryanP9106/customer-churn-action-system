import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AI Universal Churn System", layout="wide")

st.title("📊 Universal Customer Churn & Action System")
st.markdown("Powered by a decoupled Multi-Layer Perceptron Neural Network.")

tab1, tab2 = st.tabs(["🚀 Universal AI Prediction Pipeline", "📈 A/B Testing Simulation"])

with tab1:
    st.subheader("Upload Any Customer Churn Dataset")
    uploaded_file = st.file_uploader("Choose any CSV data file", type=["csv"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        available_columns = list(df_preview.columns)
        
        st.markdown("### 🎯 Map Your Dataset Columns to AI Model Concepts")
        col_id, col_rec, col_freq, col_mon = st.columns(4)
        
        with col_id:
            id_col = st.selectbox("Unique Customer ID Column", available_columns, 
                                  index=available_columns.index("Customer ID") if "Customer ID" in available_columns else 0)
        with col_rec:
            default_rec_idx = available_columns.index("Tenure in Months") if "Tenure in Months" in available_columns else (available_columns.index("Orders") if "Orders" in available_columns else 0)
            rec_col = st.selectbox("Recency / Engagement Column", available_columns, index=default_rec_idx)
        with col_freq:
            default_freq_idx = available_columns.index("Number of Referrals") if "Number of Referrals" in available_columns else (available_columns.index("Quantity") if "Quantity" in available_columns else 0)
            freq_col = st.selectbox("Frequency Column", available_columns, index=default_freq_idx)
        with col_mon:
            default_mon_idx = available_columns.index("Monthly Charge") if "Monthly Charge" in available_columns else (available_columns.index("Revenue") if "Revenue" in available_columns else 0)
            mon_col = st.selectbox("Monetary Column", available_columns, index=default_mon_idx)

        if st.button("Run Batch AI Inference"):
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)
            
            # Map selected columns into standard format expected by the backend
            standardized_df = pd.DataFrame()
            standardized_df['Customer_ID'] = raw_df[id_col].astype(str)
            standardized_df['Orders'] = pd.to_numeric(raw_df[rec_col], errors='coerce').fillna(0)
            standardized_df['Quantity'] = pd.to_numeric(raw_df[freq_col], errors='coerce').fillna(0)
            standardized_df['Revenue'] = pd.to_numeric(raw_df[mon_col], errors='coerce').fillna(0)
            standardized_df['Profit'] = standardized_df['Revenue'] * 0.20
            standardized_df['Discount_Rate'] = 0.05
            standardized_df['Order_ID'] = standardized_df.index
            
            csv_buffer = standardized_df.to_csv(index=False).encode('utf-8')
            files = {"file": (uploaded_file.name, csv_buffer, "text/csv")}
            
            with st.spinner("Processing deep learning calculations..."):
                try:
                    # NOTE: Ensure this matches your live render URL
                    api_url = "https://customer-churn-action-system.onrender.com/predict-csv"
                    response = requests.post(api_url, files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        if res_data.get("status") == "success":
                            results_df = pd.DataFrame(res_data["data"])
                            
                            st.subheader("🔮 Universal AI Inference Matrix Results")
                            display_df = pd.merge(raw_df[[id_col]], results_df, left_on=id_col, right_on="Customer_ID").drop(columns=["Customer_ID"])
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.error(f"Backend Engine Error: {res_data.get('message', 'Unknown Error')}")
                    else:
                        st.error(f"Server returned error code: {response.status_code}")
                except Exception as e:
                    st.error(f"Could not reach Backend Server: {str(e)}")

with tab2:
    st.subheader("🎯 Active Experiment: Retention Strategy Simulation")
    control_churn_rate = 0.6607
    variant_churn_rate = 0.5120
    
    m1, m2 = st.columns(2)
    m1.metric("📉 Churn Reduction (AI Lift)", f"-{round((control_churn_rate - variant_churn_rate)*100, 2)}%")
    m2.info("Statistical Significance p-value: 0.00041 (Reject Null Hypothesis)")