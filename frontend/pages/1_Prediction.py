import streamlit as st
import pandas as pd
import requests

COLUMN_ALIASES = {
    "Customer_ID": [
        "Customer_ID",
        "Customer ID",
        "CustomerID",
        "Customer Id",
        "customer_id",
        "id"
    ],
    "Orders": [
        "Orders",
        "Order Count",
        "Order_Count",
        "Tenure in Months",
        "Tenure",
        "Recency",
        "Engagement"
    ],
    "Quantity": [
        "Quantity",
        "Qty",
        "Number of Referrals",
        "Referrals",
        "Frequency",
        "Purchase Frequency"
    ],
    "Revenue": [
        "Revenue",
        "Total Revenue",
        "Monthly Charge",
        "Monthly Charges",
        "Total Charges",
        "Sales",
        "Amount"
    ],
    "Profit": [
        "Profit",
        "Gross Profit",
        "Margin"
    ],
    "Discount_Rate": [
        "Discount_Rate",
        "Discount Rate",
        "Discount"
    ]
}


def normalize_column_name(column_name):
    return "".join(
        char.lower()
        for char in str(column_name)
        if char.isalnum()
    )


def find_column(df, aliases):

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for alias in aliases:

        match = normalized_columns.get(
            normalize_column_name(alias)
        )

        if match is not None:
            return match

    return None


def standardize_for_prediction(raw_df):

    standardized_df = pd.DataFrame(index=raw_df.index)

    source_columns = {
        target: find_column(raw_df, aliases)
        for target, aliases in COLUMN_ALIASES.items()
    }

    id_col = source_columns["Customer_ID"]

    standardized_df["Customer_ID"] = (
        raw_df[id_col].astype(str)
        if id_col is not None
        else raw_df.index.astype(str)
    )

    for target in ["Orders", "Quantity", "Revenue"]:

        source = source_columns[target]

        standardized_df[target] = (
            pd.to_numeric(
                raw_df[source],
                errors="coerce"
            ).fillna(0)
            if source is not None
            else 0
        )

    profit_col = source_columns["Profit"]

    standardized_df["Profit"] = (
        pd.to_numeric(
            raw_df[profit_col],
            errors="coerce"
        ).fillna(0)
        if profit_col is not None
        else standardized_df["Revenue"] * 0.20
    )

    discount_col = source_columns["Discount_Rate"]

    standardized_df["Discount_Rate"] = (
        pd.to_numeric(
            raw_df[discount_col],
            errors="coerce"
        ).fillna(0)
        if discount_col is not None
        else 0.05
    )

    return standardized_df


st.title("Prediction Pipeline")

if "raw_df" not in st.session_state:

    st.info("Upload a dataset from the Home page.")
    st.stop()

raw_df = st.session_state["raw_df"]

st.subheader("Dataset Preview")

st.dataframe(
    raw_df.head(10),
    use_container_width=True
)

if st.button("Run Batch AI Inference"):

    standardized_df = standardize_for_prediction(raw_df)

    csv_buffer = (
        standardized_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    files = {
        "file": (
            "dataset.csv",
            csv_buffer,
            "text/csv"
        )
    }

    with st.spinner("Running AI inference..."):

        response = requests.post(
            "https://customer-churn-action-system.onrender.com/predict-csv",
            files=files
        )

    if response.status_code == 200:

        data = response.json()

        if data["status"] == "success":

            results_df = pd.DataFrame(
                data["data"]
            )

            st.session_state["results_df"] = results_df

            display_df = pd.concat(
                [
                    raw_df.reset_index(drop=True),
                    results_df.drop(
                        columns=["Customer_ID"]
                    ).reset_index(drop=True)
                ],
                axis=1
            )

            st.success(
                "Inference completed successfully."
            )

            st.dataframe(
                display_df,
                use_container_width=True
            )

            st.info("Proceed to the A/B Testing and Revenue At Risk modules using the sidebar.")

        else:
            st.error(data["message"])

    else:
        st.error(
            f"Server Error: {response.status_code}"
        )

    