import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import chi2_contingency

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
    "Order_ID": [
        "Order_ID",
        "Order ID",
        "OrderID",
    ],
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
        target_column: find_column(raw_df, aliases)
        for target_column, aliases in COLUMN_ALIASES.items()
    }

    id_col = source_columns["Customer_ID"]

    standardized_df["Customer_ID"] = (
        raw_df[id_col].astype(str)
        if id_col is not None
        else raw_df.index.astype(str)
    )

    for target_column in [
        "Orders",
        "Quantity",
        "Revenue",
    ]:
        source_col = source_columns[target_column]

        standardized_df[target_column] = (
            pd.to_numeric(
                raw_df[source_col],
                errors="coerce"
            ).fillna(0.0)
            if source_col is not None
            else 0.0
        )

    profit_col = source_columns["Profit"]

    standardized_df["Profit"] = (
        pd.to_numeric(
            raw_df[profit_col],
            errors="coerce"
        ).fillna(0.0)
        if profit_col is not None
        else standardized_df["Revenue"] * 0.20
    )

    discount_col = source_columns["Discount_Rate"]

    standardized_df["Discount_Rate"] = (
        pd.to_numeric(
            raw_df[discount_col],
            errors="coerce"
        ).fillna(0.0)
        if discount_col is not None
        else 0.05
    )

    order_col = source_columns["Order_ID"]

    standardized_df["Order_ID"] = (
        raw_df[order_col].astype(str)
        if order_col is not None
        else raw_df.index.astype(str)
    )

    detected_columns = {
        target: source
        for target, source in source_columns.items()
        if source is not None
    }

    return standardized_df, detected_columns


st.set_page_config(
    page_title="AI Universal Churn System",
    layout="wide"
)

st.title("AI Universal Customer Churn & Action System")

st.markdown(
    "Powered by a Multi-Layer Perceptron Neural Network."
)

tab1, tab2, tab3 = st.tabs(
    [
        "Universal AI Prediction Pipeline",
        "A/B Testing Simulation",
        "Revenue At Risk"
    ]
)

with tab1:

    st.subheader("Upload Customer Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV, XLSX or TSV file",
        type=["csv", "xlsx", "tsv"]
    )

    if uploaded_file is not None:

        st.success("File uploaded successfully!")

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

        else:
            st.error("Unsupported file format.")
            st.stop()

        raw_df.columns = (
            raw_df.columns
            .str.strip()
            .str.replace(" ", "_")
        )

        raw_df = raw_df.drop_duplicates()
        raw_df = raw_df.fillna(0)

        standardized_df, detected_columns = (
            standardize_for_prediction(raw_df)
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            len(raw_df)
        )

        c2.metric(
            "Columns",
            len(raw_df.columns)
        )

        c3.metric(
            "Missing Values",
            raw_df.isna().sum().sum()
        )

        if detected_columns:

            st.caption(
                "Detected model inputs: "
                + ", ".join(
                    f"{target} ← {source}"
                    for target, source
                    in detected_columns.items()
                )
            )

        st.dataframe(
            raw_df.head(10),
            use_container_width=True
        )

        if st.button("Run Batch AI Inference"):

            csv_buffer = (
                standardized_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            files = {
                "file": (
                    uploaded_file.name,
                    csv_buffer,
                    "text/csv"
                )
            }

            with st.spinner(
                "Processing deep learning calculations..."
            ):

                try:

                    api_url = (
                        "https://customer-churn-action-system.onrender.com/predict-csv"
                    )

                    response = requests.post(
                        api_url,
                        files=files
                    )

                    if response.status_code == 200:

                        res_data = response.json()

                        if (
                            res_data.get("status")
                            == "success"
                        ):

                            results_df = pd.DataFrame(
                                res_data["data"]
                            )

                            st.session_state[
                                "results_df"
                            ] = results_df

                            display_df = pd.concat(
                                [
                                    raw_df.reset_index(
                                        drop=True
                                    ),
                                    results_df.drop(
                                        columns=[
                                            "Customer_ID"
                                        ]
                                    ).reset_index(
                                        drop=True
                                    ),
                                ],
                                axis=1,
                            )

                            st.subheader(
                                "Universal AI Inference Results"
                            )

                            st.dataframe(
                                display_df,
                                use_container_width=True
                            )

                        else:
                            st.error(
                                res_data.get(
                                    "message"
                                )
                            )

                    else:
                        st.error(
                            f"Server Error: {response.status_code}"
                        )

                except Exception as e:
                    st.error(
                        f"Backend Error: {str(e)}"
                    )

with tab2:

    st.subheader(
        "AI Retention A/B Testing Simulation"
    )

    if "results_df" not in st.session_state:

        st.info("Run AI inference first.")

    else:

        results_df = st.session_state["results_df"].copy()

        results_df["Revenue"] = pd.to_numeric(
            results_df["Revenue"],
            errors="coerce"
        ).fillna(0)

        results_df["Revenue_at_Risk"] = pd.to_numeric(
            results_df["Revenue_at_Risk"],
            errors="coerce"
        ).fillna(0)

        np.random.seed(42)

        results_df["Group"] = np.random.choice(
            ["Control", "Treatment"],
            size=len(results_df)
        )

        results_df["Churned"] = (
            results_df["Risk_Tier"] == "High"
        ).astype(int)

        treatment_effect = 0.25

        treatment_mask = (
            (results_df["Group"] == "Treatment")
            &
            (results_df["Risk_Tier"] == "High")
        )

        results_df.loc[
            treatment_mask,
            "Churned"
        ] = np.random.binomial(
            1,
            1 - treatment_effect,
            size=treatment_mask.sum()
        )

        control_rate = (
            results_df[
                results_df["Group"] == "Control"
            ]["Churned"].mean()
        )

        treatment_rate = (
            results_df[
                results_df["Group"] == "Treatment"
            ]["Churned"].mean()
        )

        lift = (
            (control_rate - treatment_rate)
            * 100
        )

        contingency = pd.crosstab(
            results_df["Group"],
            results_df["Churned"]
        )

        chi2, p_value, _, _ = chi2_contingency(
            contingency
        )

        high_risk_customers = results_df[
            results_df["Risk_Tier"] == "High"
        ]

        recovered_revenue = (
            high_risk_customers[
                "Revenue_at_Risk"
            ].sum()
            * treatment_effect
        )

        saved_customers = (
            (control_rate - treatment_rate)
            * len(results_df)
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Control Churn",
            f"{control_rate*100:.2f}%"
        )

        c2.metric(
            "Treatment Churn",
            f"{treatment_rate*100:.2f}%"
        )

        c3.metric(
            "AI Lift",
            f"{lift:.2f}%"
        )

        if p_value < 0.05:

            st.success(
                f"Statistically Significant (p={p_value:.4f})"
            )

        else:

            st.warning(
                f"Not Significant (p={p_value:.4f})"
            )

        st.info(
            f"""
💰 AI Retention Impact

Estimated revenue preserved:
₹{recovered_revenue:,.0f}

Approximately {saved_customers:.1f}
customers were retained through the
AI-driven intervention strategy.
"""
        )

        st.subheader("Experiment Groups")

        st.dataframe(
            results_df[
                [
                    "Customer_ID",
                    "Risk_Tier",
                    "Group",
                    "Churned"
                ]
            ],
            use_container_width=True
        )

        st.subheader(
            "Control vs Treatment Churn"
        )

        st.bar_chart(
            pd.Series(
                {
                    "Control": control_rate,
                    "Treatment": treatment_rate
                }
            )
        )

with tab3:

    st.subheader("Revenue At Risk Dashboard")

    if "results_df" not in st.session_state:

        st.info("Run AI inference first.")

    else:

        results_df = st.session_state["results_df"]


        if "Revenue_at_Risk" not in results_df.columns:
            st.error(
                "Revenue_at_Risk column was not returned by the backend."
            )

            st.write("Backend returned:")
            st.dataframe(results_df)

        else:

            results_df["Revenue_at_Risk"] = pd.to_numeric(
                results_df["Revenue_at_Risk"],
                errors="coerce"
            ).fillna(0)

            total_risk = (
                results_df["Revenue_at_Risk"]
                .sum()
            )

            high_risk = (
                results_df[
                    results_df["Risk_Tier"] == "High"
                ]["Revenue_at_Risk"]
                .sum()
            )

            medium_risk = (
                results_df[
                    results_df["Risk_Tier"] == "Medium"
                ]["Revenue_at_Risk"]
                .sum()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Revenue At Risk",
                f"₹{total_risk:,.2f}"
            )

            c2.metric(
                "High Risk Revenue",
                f"₹{high_risk:,.2f}"
            )

            c3.metric(
                "Medium Risk Revenue",
                f"₹{medium_risk:,.2f}"
            )

            st.subheader(
                "Top Financially Risky Customers"
            )

            top_customers = (
                results_df
                .sort_values(
                    "Revenue_at_Risk",
                    ascending=False
                )
            )

            st.dataframe(
                top_customers,
                use_container_width=True
            )

            st.subheader(
                "Revenue Exposure by Risk Tier"
            )

            risk_chart = (
                results_df
                .groupby("Risk_Tier")
                ["Revenue_at_Risk"]
                .sum()
            )

            st.bar_chart(risk_chart)

            if total_risk > 100000:

                st.error(
                    f"Estimated revenue exposure: ₹{total_risk:,.0f}"
                )

            else:

                st.success(
                    "Revenue exposure is manageable."
                )