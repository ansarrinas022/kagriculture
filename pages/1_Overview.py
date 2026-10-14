"""
Page 1: Overview
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd

from config import APP_TITLE, CUSTOM_CSS
from src.state_manager import render_sidebar_status_and_filters
from src.episode_analyzer import EpisodeAnalyzer
from src.action_analyzer import ActionAnalyzer
from src.visualization import Visualizer

st.set_page_config(page_title=f"Overview | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("📊 Simulation Overview & Metrics")
st.markdown("<p style='color:#9CA3AF;'>Comprehensive macro metrics and distribution profiles for the active episode cohort.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No episodes match the selected global filters. Please adjust your sidebar settings.")
else:
    analyzer = EpisodeAnalyzer(filtered_df)
    metrics = analyzer.get_summary_metrics()
    action_analyzer = ActionAnalyzer(filtered_df, filtered_actions_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Episodes", f"{metrics['total_episodes']:,}")
    col2.metric("Unique Agents", f"{metrics['unique_agents_count']}")
    col3.metric("Average Score", f"${metrics['avg_score']:,.2f}" if metrics['avg_score'] is not None else "N/A")
    col4.metric("Highest Score", f"${metrics['max_score']:,.2f}" if metrics['max_score'] is not None else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Minimum Score", f"${metrics['min_score']:,.2f}" if metrics['min_score'] is not None else "N/A")
    col6.metric("Total Actions Recorded", f"{metrics['total_actions']:,}" if metrics['total_actions'] is not None else "N/A")
    col7.metric("Memory Usage", f"{(metrics['dataset_size_bytes'] / 1024):,.1f} KB")
    col8.metric("Valid Records", f"{metrics['valid_records_count']} / {metrics['total_episodes']}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(Visualizer.plot_score_distribution(filtered_df), use_container_width=True)
    with c2:
        leaderboard = analyzer.get_agent_leaderboard()
        st.plotly_chart(Visualizer.plot_agent_leaderboard_bar(leaderboard), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        act_freq = action_analyzer.get_action_frequencies()
        st.plotly_chart(Visualizer.plot_action_distribution(act_freq), use_container_width=True)
    with c4:
        st.plotly_chart(Visualizer.plot_performance_trends(filtered_df), use_container_width=True)

    if metrics.get("win_rate_available"):
        st.plotly_chart(Visualizer.plot_outcome_distribution(filtered_df), use_container_width=True)

    st.markdown("### 📋 Cohort Data Preview")
    st.dataframe(filtered_df.head(25), use_container_width=True)
