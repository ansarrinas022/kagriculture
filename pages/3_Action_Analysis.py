"""
Page 3: Farming Action Analysis
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import APP_TITLE, CUSTOM_CSS, THEME_COLORS
from src.state_manager import render_sidebar_status_and_filters
from src.action_analyzer import ActionAnalyzer
from src.visualization import Visualizer, PLOTLY_LAYOUT_DEFAULTS

st.set_page_config(page_title=f"Action Analysis | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("🚜 Farming Action & Operational Analysis")
st.markdown("<p style='color:#9CA3AF;'>Evaluate turn-level farming actions: planting cycles, watering frequencies, livestock care, and commercial trading.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No simulation data found matching current filters.")
else:
    analyzer = ActionAnalyzer(filtered_df, filtered_actions_df)
    action_freq = analyzer.get_action_frequencies()

    if action_freq.empty:
        st.info("No action data detected in current records.")
    else:
        # Top KPI cards for actions
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        top_act = action_freq.iloc[0]["action"] if not action_freq.empty else "N/A"
        tot_ops = action_freq["count"].sum()
        with kpi1:
            st.metric("Total Recorded Operations", f"{tot_ops:,}")
        with kpi2:
            st.metric("Distinct Action Types", f"{len(action_freq)}")
        with kpi3:
            st.metric("Dominant Action", top_act)
        with kpi4:
            dom_pct = action_freq.iloc[0]["percentage"] if not action_freq.empty else 0
            st.metric("Dominant Share", f"{dom_pct}%")

        st.markdown("---")

        # Action distribution charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(Visualizer.plot_action_distribution(action_freq), use_container_width=True)
        with c2:
            # Bar chart by category
            cat_summary = action_freq.groupby("category")["count"].sum().reset_index().sort_values(by="count", ascending=True)
            fig_cat = go.Figure(go.Bar(
                x=cat_summary["count"],
                y=cat_summary["category"],
                orientation="h",
                marker_color="#10B981"
            ))
            fig_cat.update_layout(
                title="<b>Operations by Farming Category</b>",
                xaxis_title="Action Count",
                yaxis_title="",
                **PLOTLY_LAYOUT_DEFAULTS
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Winning vs Losing Action Comparison
        st.markdown("### 🏆 **Action Profiles: Winning vs Losing Matches**")
        st.markdown("<span style='color:#9CA3AF;'>Identifies operational behaviors that correlate with positive match outcomes.</span>", unsafe_allow_html=True)
        
        comp_df = analyzer.compare_winning_vs_losing_actions()
        if not comp_df.empty:
            if "Difference (% pts)" in comp_df.columns:
                c_fig = px.bar(
                    comp_df,
                    x="Difference (% pts)",
                    y="action",
                    orientation="h",
                    color="Difference (% pts)",
                    color_continuous_scale=["#EF4444", "#34D399", "#10B981"],
                    title="<b>Net Percentage Point Advantage for Winners by Action</b>"
                )
                c_fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                st.plotly_chart(c_fig, use_container_width=True)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Match outcome labels are required to compare winning vs losing operations.")

        # Resource Management & Scores Relationship
        st.markdown("### 🌾 **Resource Conversions & Score Elasticity**")
        if "crops_harvested_a" in filtered_df.columns and "score_a" in filtered_df.columns:
            # Scatter plot of harvests vs final score
            fig_scatter = px.scatter(
                filtered_df,
                x="crops_harvested_a",
                y="score_a",
                color="agent_a" if "agent_a" in filtered_df.columns else None,
                trendline="ols",
                title="<b>Crop Harvest Volume vs Final Bank Balance</b>",
                labels={"crops_harvested_a": "Crops Harvested (Agent A)", "score_a": "Final Score ($)"}
            )
            fig_scatter.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Action by Agent Cross-Tabulation
        st.markdown("### 👥 **Farming Actions by Competing Agent**")
        cross = analyzer.get_actions_by_agent()
        if not cross.empty:
            st.dataframe(cross, use_container_width=True)
