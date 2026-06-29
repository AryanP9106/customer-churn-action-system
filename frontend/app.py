import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Churn-Action System",
    page_icon="📈",
    layout="wide"
)

st.title("Customer Churn & Action System")

st.markdown("""
Welcome to the AI-powered customer churn analytics platform.

Use the sidebar to navigate between:

- Prediction Pipeline
- A/B Testing
- Revenue At Risk
""")

st.sidebar.success("Select a module above.")

st.subheader("Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV, XLSX or TSV file",
    type=["csv", "xlsx", "tsv"]
)

if uploaded_file is not None:

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)

    elif filename.endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded_file)

    elif filename.endswith(".tsv"):
        raw_df = pd.read_csv(
            uploaded_file,
            sep="\t"
        )

    raw_df.columns = (
        raw_df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    raw_df = raw_df.drop_duplicates()
    raw_df = raw_df.fillna(0)

    st.session_state["raw_df"] = raw_df

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(raw_df))
    c2.metric("Columns", len(raw_df.columns))
    c3.metric(
        "Missing Values",
        raw_df.isna().sum().sum()
    )

    st.success("Dataset uploaded successfully.")

    st.dataframe(
        raw_df.head(),
        use_container_width=True
    )

    st.info(
        "Use the sidebar to continue to Prediction."
    )