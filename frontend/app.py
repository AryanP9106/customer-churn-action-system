import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats

st.set_page_config(layout="wide")
st.title("📊 Enterprise Customer Churn & Experimentation Platform")

tab1, tab2 = st.tabs(["🚀 Bulk AI Prediction Pipeline", "🔬 Live A/B Testing Simulator"])

with tab1:
    st.header("Upload Customer Cohort")
    uploaded_file = st.file_uploader("Upload client data (CSV format)", type="csv")
    
    if uploaded_file is not None:
        if st.button("Run Batch AI Inference"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            
            with st.spinner("Processing data through decoupled backend..."):
                try:
                    # Point to local FastAPI instance
                    response = requests.post("http://127.0.0.1:8000/predict-csv", files=files)
                    if response.status_code == 200:
                        output = response.json()
                        if output["status"] == "success":
                            res_df = pd.DataFrame(output["data"])
                            st.success("Analysis complete!")
                            st.dataframe(res_df, use_container_width=True)
                        else:
                            st.error(output["message"])
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend. Ensure it is running on port 8000.")

with tab2:
    st.header("Campaign Experimentation Framework")
    st.write("Simulate if your AI-driven recommendations statistically outperform baseline offers.")
    
    total_users = st.slider("Simulation Sample Size (Total Customers)", 100, 5000, 1000, 100)
    
    if st.button("Run Statistical Chi-Square Test"):
        half = total_users // 2
        
        # Simulating outcomes
        group_a_churn = np.random.binomial(half, 0.22)
        group_b_churn = np.random.binomial(half, 0.13)
        
        contingency_table = [
            [half - group_a_churn, group_a_churn],
            [half - group_b_churn, group_b_churn]
        ]
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Control Group Churn (Standard Offer)", f"{(group_a_churn/half):.1%}")
        col2.metric("Variant Group Churn (AI Targeted)", f"{(group_b_churn/half):.1%}")
        col3.metric("P-Value", f"{p_value:.5f}")
        
        if p_value < 0.05:
            st.success("✨ Statistically Significant: The AI system effectively minimized user churn.")
        else:
            st.warning("⚠️ Insufficient Variance: The churn difference is minor; collect more metrics.")