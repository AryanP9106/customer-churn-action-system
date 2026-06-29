import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

st.title("A/B Testing Simulation")

if "results_df" not in st.session_state:

    st.info("Run AI inference first.")

else:

    results_df = (
        st.session_state["results_df"]
        .copy()
    )

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

    mask = (
        (results_df["Group"] == "Treatment")
        &
        (results_df["Risk_Tier"] == "High")
    )

    results_df.loc[
        mask,
        "Churned"
    ] = np.random.binomial(
        1,
        1 - treatment_effect,
        size=mask.sum()
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
        control_rate - treatment_rate
    ) * 100

    contingency = pd.crosstab(
        results_df["Group"],
        results_df["Churned"]
    )

    chi2, p_value, _, _ = (
        chi2_contingency(contingency)
    )

    recovered_revenue = (
        results_df[
            results_df["Risk_Tier"] == "High"
        ]["Revenue_at_Risk"]
        .sum()
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
💰 Estimated revenue preserved:

₹{recovered_revenue:,.0f}

Approximately {saved_customers:.1f}
customers were retained.
"""
    )

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

    st.bar_chart(
        pd.Series(
            {
                "Control": control_rate,
                "Treatment": treatment_rate
            }
        )
    )

    st.info("Proceed to view the Revenue at Risk and other modules.")