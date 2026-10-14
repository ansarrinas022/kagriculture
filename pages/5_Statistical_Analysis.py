"""
Page 5: Statistical Analysis
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd
import numpy as np

from config import APP_TITLE, CUSTOM_CSS
from src.state_manager import render_sidebar_status_and_filters
from src.statistical_analysis import StatisticalAnalysis
from src.visualization import Visualizer

st.set_page_config(page_title=f"Statistical Analysis | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("📈 Statistical Analysis & Rigorous Distributions")
st.markdown("<p style='color:#9CA3AF;'>Perform descriptive audits, correlation matrices, outlier diagnostics, and subgroup analyses with statistical safeguards.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No data available to perform statistical analysis.")
else:
    stats = StatisticalAnalysis(filtered_df)
    num_cols = stats.get_numerical_columns()

    # Tabs for organized statistical navigation
    tab_desc, tab_corr, tab_outlier, tab_group = st.tabs([
        "📊 Descriptive Statistics",
        "🔗 Correlation Heatmap",
        "🚨 Outlier Detection",
        "📑 Grouped Aggregations"
    ])

    with tab_desc:
        st.markdown("### 📋 **Parametric & Non-Parametric Feature Summary**")
        selected_cols = st.multiselect(
            "Select Numerical Features to Audit",
            options=num_cols,
            default=num_cols[:min(6, len(num_cols))]
        )

        if selected_cols:
            desc_df = stats.compute_descriptive_stats(selected_cols)
            st.dataframe(
                desc_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Select at least one numerical feature above.")

    with tab_corr:
        st.markdown("### 🔗 **Pairwise Correlation Matrix**")
        st.markdown("""
        <div style="background:rgba(217, 119, 6, 0.15); border:1px solid #D97706; padding:10px 14px; border-radius:6px; margin-bottom:14px;">
            <span style="color:#FBBF24; font-weight:600;">⚠️ Critical Scientific Principle:</span>
            <span style="color:#ECFDF5; font-size:0.9rem;"> Statistical correlation measures mathematical co-movement and does <b>NOT</b> imply causality. Observed correlations may be mediated by latent game factors such as early season RNG.</span>
        </div>
        """, unsafe_allow_html=True)

        c_col1, c_col2 = st.columns([3, 1])
        with c_col1:
            corr_feature_selection = st.multiselect(
                "Select Features for Correlation",
                options=num_cols,
                default=num_cols[:min(6, len(num_cols))]
            )
        with c_col2:
            corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"])

        if len(corr_feature_selection) >= 2:
            corr_matrix, note = stats.compute_correlation_matrix(corr_feature_selection, method=corr_method)
            if not corr_matrix.empty:
                st.plotly_chart(Visualizer.plot_correlation_heatmap(corr_matrix), use_container_width=True)
                st.dataframe(corr_matrix, use_container_width=True)
            else:
                st.warning(note)
        else:
            st.info("Select at least two numerical columns to generate correlation heatmaps.")

    with tab_outlier:
        st.markdown("### 🚨 **Outlier Detection Suite**")
        st.markdown("<span style='color:#9CA3AF;'>Detect unusual or extreme matches that deviate significantly from standard simulation baselines.</span>", unsafe_allow_html=True)

        o_col1, o_col2, o_col3 = st.columns(3)
        with o_col1:
            outlier_col = st.selectbox("Feature to Analyze for Outliers", options=num_cols)
        with o_col2:
            outlier_method = st.selectbox("Detection Method", ["IQR", "Z-Score"])
        with o_col3:
            threshold = st.slider(
                "Threshold Multiplier",
                min_value=1.0,
                max_value=3.5,
                value=1.5 if outlier_method == "IQR" else 2.5,
                step=0.1
            )

        outlier_res = stats.detect_outliers(outlier_col, method=outlier_method.lower(), threshold=threshold)
        if "error" in outlier_res:
            st.warning(outlier_res["error"])
        else:
            stat_c1, stat_c2, stat_c3 = st.columns(3)
            stat_c1.metric("Outlier Episodes Detected", f"{outlier_res['total_outliers']}")
            stat_c2.metric("Outlier Proportion", f"{outlier_res['outlier_percentage']}%")
            bounds_str = str(outlier_res['bounds'])
            stat_c3.metric("Calculation Bounds", bounds_str[:25] + "...")

            if outlier_res["total_outliers"] > 0:
                st.markdown("#### 🚩 **Flagged Outlier Records**")
                st.dataframe(outlier_res["outlier_rows"], use_container_width=True)
            else:
                st.success(f"No statistical outliers detected for '{outlier_col}' at threshold {threshold}.")

    with tab_group:
        st.markdown("### 📑 **Dynamic Subgroup Grouping & Cross-Tabs**")
        cat_candidates = [c for c in filtered_df.columns if filtered_df[c].nunique() < 25 and filtered_df[c].dtype == "object"]
        if not cat_candidates and "winner" in filtered_df.columns:
            cat_candidates = ["winner", "agent_a", "agent_b"]

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            group_by = st.selectbox("Group By Categorical Variable", options=cat_candidates or ["agent_a"])
        with g_col2:
            num_target = st.selectbox("Aggregate Target Feature", options=num_cols)

        if group_by and num_target:
            grouped_df = stats.compute_grouped_statistics(group_by, num_target)
            st.dataframe(grouped_df, use_container_width=True, hide_index=True)
