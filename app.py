"""
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
Main Application Entry Point & Overview Dashboard
"""
import streamlit as st
import pandas as pd

from config import APP_TITLE, APP_SUBTITLE, APP_ICON, CUSTOM_CSS
from src.state_manager import initialize_session_state, render_sidebar_status_and_filters
from src.episode_analyzer import EpisodeAnalyzer
from src.action_analyzer import ActionAnalyzer
from src.visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Agricultural Simulation Platform",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Theme CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State & Sidebar Filters
filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

# Top Header Bar
col_logo, col_desc = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3.5rem; text-align:center;'>🌱</div>", unsafe_allow_html=True)
with col_desc:
    st.markdown(f"<h1 style='margin:0; color:#10B981; font-weight:800; letter-spacing:-0.02em;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#9CA3AF; margin:0 0 12px 0; font-size:1.05rem;'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)

st.markdown("---")

# Main Content: Page A - Overview
st.markdown("## 📊 **Simulation Overview & Performance Matrix**")
st.markdown("<span style='color:#9CA3AF;'>Real-time metrics computed directly from loaded Kaggriculture episodes.</span>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No episodes match the selected global filters. Please adjust the sidebar filter controls or reset.")
else:
    analyzer = EpisodeAnalyzer(filtered_df)
    metrics = analyzer.get_summary_metrics()
    action_analyzer = ActionAnalyzer(filtered_df, filtered_actions_df)

    # Metric Cards Row 1
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="Total Episodes",
            value=f"{metrics['total_episodes']:,}",
            help="Total simulation matches matching current filters."
        )
    with m_col2:
        st.metric(
            label="Competing Agents",
            value=f"{metrics['unique_agents_count']}",
            help="Distinct autonomous farming agents detected in the dataset."
        )
    with m_col3:
        avg_s = f"${metrics['avg_score']:,.2f}" if metrics['avg_score'] is not None else "N/A"
        st.metric(
            label="Average Score",
            value=avg_s,
            help="Mean final bank balance achieved by competing agents."
        )
    with m_col4:
        max_s = f"${metrics['max_score']:,.2f}" if metrics['max_score'] is not None else "N/A"
        st.metric(
            label="Peak Episode Score",
            value=max_s,
            help="Highest recorded bank balance achieved in a single match."
        )

    # Metric Cards Row 2
    m_col5, m_col6, m_col7, m_col8 = st.columns(4)
    with m_col5:
        min_s = f"${metrics['min_score']:,.2f}" if metrics['min_score'] is not None else "N/A"
        st.metric(
            label="Lowest Score",
            value=min_s,
            help="Minimum bank balance recorded across analyzed episodes."
        )
    with m_col6:
        tot_act = f"{metrics['total_actions']:,}" if metrics['total_actions'] is not None else "N/A"
        st.metric(
            label="Recorded Actions",
            value=tot_act,
            help="Total farming and management operations recorded."
        )
    with m_col7:
        mem_kb = f"{(metrics['dataset_size_bytes'] / 1024):,.1f} KB"
        st.metric(
            label="Memory Footprint",
            value=mem_kb,
            help="Total in-memory DataFrame storage used."
        )
    with m_col8:
        val_status = "100% Valid" if metrics['valid_records_count'] == metrics['total_episodes'] else f"{metrics['valid_records_count']}/{metrics['total_episodes']}"
        st.metric(
            label="Record Integrity",
            value=val_status,
            help="Validated records containing verifiable episode IDs."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Visualizations Grid
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.plotly_chart(Visualizer.plot_score_distribution(filtered_df), use_container_width=True)
    with v_col2:
        leaderboard = analyzer.get_agent_leaderboard()
        st.plotly_chart(Visualizer.plot_agent_leaderboard_bar(leaderboard), use_container_width=True)

    v_col3, v_col4 = st.columns(2)
    with v_col3:
        act_freq = action_analyzer.get_action_frequencies()
        st.plotly_chart(Visualizer.plot_action_distribution(act_freq), use_container_width=True)
    with v_col4:
        st.plotly_chart(Visualizer.plot_performance_trends(filtered_df), use_container_width=True)

    if metrics.get("win_rate_available"):
        st.plotly_chart(Visualizer.plot_outcome_distribution(filtered_df), use_container_width=True)

    # Leaderboard Table Preview
    st.markdown("### 🏆 **Benchmark Agent Standings**")
    if not leaderboard.empty:
        st.dataframe(
            leaderboard,
            use_container_width=True,
            column_config={
                "agent": st.column_config.TextColumn("Agent Name", width="medium"),
                "episodes": st.column_config.NumberColumn("Matches Played"),
                "avg_score": st.column_config.NumberColumn("Avg Score ($)", format="$%.2f"),
                "median_score": st.column_config.NumberColumn("Median ($)", format="$%.2f"),
                "max_score": st.column_config.NumberColumn("Peak ($)", format="$%.2f"),
                "win_rate": st.column_config.ProgressColumn("Win Rate (%)", format="%.1f%%", min_value=0, max_value=100),
                "avg_harvests": st.column_config.NumberColumn("Avg Harvests"),
                "avg_market_sales": st.column_config.NumberColumn("Avg Sales ($)", format="$%.2f")
            },
            hide_index=True
        )

# Module Shortcuts Footer
st.markdown("---")
st.markdown("### 🧭 **Platform Navigation Modules**")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown("""
    <div class="kag-card">
        <h4 style="margin:0 0 8px 0; color:#34D399;">🔍 Episode Explorer</h4>
        <p style="color:#9CA3AF; font-size:0.85rem; margin:0;">Browse step-by-step turns, verify replays, and inspect agent action histories.</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div class="kag-card">
        <h4 style="margin:0 0 8px 0; color:#34D399;">🚜 Farming Actions</h4>
        <p style="color:#9CA3AF; font-size:0.85rem; margin:0;">Compare planting, harvesting, watering, livestock, and market trade distributions.</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <div class="kag-card">
        <h4 style="margin:0 0 8px 0; color:#34D399;">⚔️ Agent Comparison</h4>
        <p style="color:#9CA3AF; font-size:0.85rem; margin:0;">Head-to-head match analytics, score differential radar, and direct rivalry stats.</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col4:
    st.markdown("""
    <div class="kag-card">
        <h4 style="margin:0 0 8px 0; color:#34D399;">💡 Strategy & ML</h4>
        <p style="color:#9CA3AF; font-size:0.85rem; margin:0;">Unsupervised strategy clustering, score prediction models, and evidence-based rules.</p>
    </div>
    """, unsafe_allow_html=True)
