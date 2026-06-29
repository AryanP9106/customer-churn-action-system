import streamlit as st
import pandas as pd

st.title("Revenue At Risk Dashboard")

if "results_df" not in st.session_state:

    st.info("Run AI inference first.")

else:

    results_df = (
        st.session_state["results_df"]
        .copy()
    )

    results_df["Revenue_at_Risk"] = (
        pd.to_numeric(
            results_df["Revenue_at_Risk"],
            errors="coerce"
        )
        .fillna(0)
    )

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
        f"₹{total_risk:,.0f}"
    )

    c2.metric(
        "High Risk Revenue",
        f"₹{high_risk:,.0f}"
    )

    c3.metric(
        "Medium Risk Revenue",
        f"₹{medium_risk:,.0f}"
    )

    st.subheader(
        "Top Risk Customers"
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_tier = st.selectbox(
            "Risk Tier",
            [
                "All",
                "High",
                "Medium",
                "Low"
            ]
        )

    filtered_df = results_df.copy()

    if selected_tier != "All":

        filtered_df = filtered_df[
            filtered_df["Risk_Tier"]
            == selected_tier
        ]

    available_sizes = [
        10,
        50,
        100,
        150,
        200
    ]

    available_sizes = [
        size
        for size in available_sizes
        if size <= len(filtered_df)
    ]

    if len(filtered_df) not in available_sizes:
        available_sizes.append(
            len(filtered_df)
        )

    available_sizes = sorted(
        available_sizes
    )

    with col2:

        sample_size = st.selectbox(
            "Sample Size",
            available_sizes
        )

    top_customers = (
        filtered_df
        .sort_values(
            "Revenue_at_Risk",
            ascending=False
        )
        .head(sample_size)
    )

    st.caption(
        f"Showing Top {len(top_customers)} "
        f"{selected_tier} risk customers "
        f"out of {len(filtered_df)} customers, "
        f"sorted by Revenue at Risk."
    )

    st.dataframe(
        top_customers,
        use_container_width=True
    )

    st.subheader(
        "Revenue Exposure"
    )

    risk_chart = (
        results_df
        .groupby("Risk_Tier")
        ["Revenue_at_Risk"]
        .sum()
    )

    st.bar_chart(risk_chart)

    if total_risk > 100000:

        st.info(
            f"Estimated exposure: "
            f"₹{total_risk:,.0f}"
        )

    else:

        st.success(
            "Revenue exposure is manageable."
        )