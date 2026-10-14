"""
Page 8: Executive Reports & Export
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from config import APP_TITLE, CUSTOM_CSS
from src.state_manager import render_sidebar_status_and_filters
from src.episode_analyzer import EpisodeAnalyzer
from src.strategy_insights import StrategyInsights
from src.agent_comparison import AgentComparison
from src.report_generator import ReportGenerator

st.set_page_config(page_title=f"Executive Reports | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("📑 Executive Reports & Briefings")
st.markdown("<p style='color:#9CA3AF;'>Generate publication-ready HTML briefs and structured CSV analytical summaries from the filtered simulation dataset.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No simulation data available to generate reports. Please adjust filters.")
else:
    analyzer = EpisodeAnalyzer(filtered_df)
    strat = StrategyInsights(filtered_df, filtered_actions_df)
    comp = AgentComparison(filtered_df, filtered_actions_df)

    summary_metrics = analyzer.get_summary_metrics()
    leaderboard = analyzer.get_agent_leaderboard()
    insights = strat.derive_evidence_based_insights()
    validation_report = st.session_state.get("validation_report", {})
    active_filters = st.session_state.get("active_filters", {})

    # Optional Head-to-Head inclusion in report
    agents = comp.get_all_agents()
    agent_comp_data = None
    if len(agents) >= 2:
        st.markdown("#### 🎯 **Optional Head-to-Head Rivalry Inclusion**")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            rep_a = st.selectbox("Report Agent A", options=agents, index=0)
        with r_col2:
            rep_b = st.selectbox("Report Agent B", options=agents, index=1 if len(agents) > 1 else 0)

        if rep_a != rep_b:
            agent_comp_data = comp.compare_agents(rep_a, rep_b)

    # Generate HTML
    html_report = ReportGenerator.generate_html_report(
        summary_metrics=summary_metrics,
        leaderboard_df=leaderboard,
        insights=insights,
        validation_report=validation_report,
        active_filters=active_filters,
        agent_comp_data=agent_comp_data
    )

    st.markdown("---")
    st.markdown("### 📥 **Download Reports**")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            label="📄 Download Formatted Executive HTML Report",
            data=html_report,
            file_name="kaggriculture_executive_report.html",
            mime="text/html",
            use_container_width=True
        )
    with d_col2:
        # Leaderboard CSV export
        lb_csv = leaderboard.to_csv(index=False)
        st.download_button(
            label="📊 Download Agent Leaderboard (CSV)",
            data=lb_csv,
            file_name="kaggriculture_agent_leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 👁️ **Live Executive Report Preview**")
    st.markdown("<span style='color:#9CA3AF; font-size:0.85rem;'>The report below can be printed directly (Ctrl+P) to save as PDF.</span>", unsafe_allow_html=True)
    components.html(html_report, height=800, scrolling=True)
