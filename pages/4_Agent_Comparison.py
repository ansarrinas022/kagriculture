"""
Page 4: Agent Performance Comparison
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import APP_TITLE, CUSTOM_CSS
from src.state_manager import render_sidebar_status_and_filters
from src.agent_comparison import AgentComparison
from src.visualization import Visualizer, PLOTLY_LAYOUT_DEFAULTS

st.set_page_config(page_title=f"Agent Comparison | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("⚔️ Head-to-Head Agent Comparison")
st.markdown("<p style='color:#9CA3AF;'>Evaluate competitive dynamics between two competing agents using matched head-to-head episodes and global benchmark performance.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No simulation episodes found matching current filters.")
else:
    comp = AgentComparison(filtered_df, filtered_actions_df)
    agents = comp.get_all_agents()

    if len(agents) < 2:
        st.warning("At least two competing agents are required to perform a comparative analysis.")
    else:
        # Agent Selection Dropdowns
        col_sel_a, col_sel_b = st.columns(2)
        with col_sel_a:
            agent_a = st.selectbox("Select Agent A", options=agents, index=0)
        with col_sel_b:
            # Default to second agent if distinct
            default_b_idx = 1 if len(agents) > 1 else 0
            agent_b = st.selectbox("Select Agent B", options=agents, index=default_b_idx)

        if agent_a == agent_b:
            st.info("Please select two distinct agents to compare head-to-head performance.")
        else:
            comp_res = comp.compare_agents(agent_a, agent_b)
            matched_stats = comp_res["matched_stats"]
            ga = comp_res["global_a"]
            gb = comp_res["global_b"]

            st.markdown("---")
            st.markdown("### 🥊 **Direct Matched Episodes (True Head-to-Head)**")
            st.markdown("<span style='color:#9CA3AF; font-size:0.85rem;'>Direct matches identify games where both agents competed in the same arena environment under identical seed conditions.</span>", unsafe_allow_html=True)

            m_count = matched_stats.get("matched_episodes_count", 0)
            if m_count > 0:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Direct Matches", f"{m_count}")
                with m2:
                    st.metric(f"{agent_a} Wins", f"{matched_stats['wins_a']} ({matched_stats['win_rate_a']}%)")
                with m3:
                    st.metric(f"{agent_b} Wins", f"{matched_stats['wins_b']} ({matched_stats['win_rate_b']}%)")
                with m4:
                    diff = matched_stats.get("avg_score_diff", 0)
                    st.metric("Avg Score Gap (A - B)", f"${diff:+,.2f}")

                # Matched Table
                matched_df = comp_res["matched_df"]
                st.markdown("#### 📋 **Direct Rivalry Match History**")
                display_cols = [c for c in ["episode_id", "score_target_a", "score_target_b", "score_diff", "winner_target", "total_turns"] if c in matched_df.columns]
                st.dataframe(
                    matched_df[display_cols].rename(columns={
                        "score_target_a": f"{agent_a} Score",
                        "score_target_b": f"{agent_b} Score",
                        "winner_target": "Winner"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(f"ℹ️ No direct matched matches were found between **{agent_a}** and **{agent_b}** in the current dataset cohort. Below is their standardized cross-arena benchmark comparison.")

            st.markdown("---")
            st.markdown("### 🌐 **Global Benchmark & Profile Comparison**")
            st.markdown("<span style='color:#9CA3AF; font-size:0.85rem;'>Performance metrics aggregated across all matches played by each agent across the dataset.</span>", unsafe_allow_html=True)

            # Metric comparison table
            comp_table = pd.DataFrame([
                {"Metric": "Total Episodes Played", agent_a: f"{ga.get('total_episodes', 0):,}", agent_b: f"{gb.get('total_episodes', 0):,}"},
                {"Metric": "Overall Win Rate (%)", agent_a: f"{ga.get('win_rate', 0)}%", agent_b: f"{gb.get('win_rate', 0)}%"},
                {"Metric": "Mean Final Score ($)", agent_a: f"${ga.get('mean_score', 0):,.2f}", agent_b: f"${gb.get('mean_score', 0):,.2f}"},
                {"Metric": "Median Final Score ($)", agent_a: f"${ga.get('median_score', 0):,.2f}", agent_b: f"${gb.get('median_score', 0):,.2f}"},
                {"Metric": "Highest Score Achieved ($)", agent_a: f"${ga.get('max_score', 0):,.2f}", agent_b: f"${gb.get('max_score', 0):,.2f}"},
                {"Metric": "Lowest Score Recorded ($)", agent_a: f"${ga.get('min_score', 0):,.2f}", agent_b: f"${gb.get('min_score', 0):,.2f}"},
                {"Metric": "Score Volatility (Std Dev $)", agent_a: f"${ga.get('std_score', 0):,.2f}", agent_b: f"${gb.get('std_score', 0):,.2f}"},
                {"Metric": "Interquartile Range (IQR $)", agent_a: f"${ga.get('iqr_score', 0):,.2f}", agent_b: f"${gb.get('iqr_score', 0):,.2f}"},
                {"Metric": "Average Crops Harvested", agent_a: f"{ga.get('avg_harvests', 0)}", agent_b: f"{gb.get('avg_harvests', 0)}"},
                {"Metric": "Average Market Sales ($)", agent_a: f"${ga.get('avg_sales', 0):,.2f}", agent_b: f"${gb.get('avg_sales', 0):,.2f}"}
            ])

            st.dataframe(comp_table, use_container_width=True, hide_index=True)

            # Radar and Box Plot Visualizations
            st.markdown("### 🕸️ **Strategic Radar & Score Stability**")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.plotly_chart(Visualizer.plot_agent_radar_comparison(comp_res), use_container_width=True)
            with rc2:
                # Box plot comparing score distributions
                fig_box = go.Figure()
                if ga.get("scores_list"):
                    fig_box.add_trace(go.Box(y=ga["scores_list"], name=agent_a, marker_color="#10B981", boxpoints="all", jitter=0.3))
                if gb.get("scores_list"):
                    fig_box.add_trace(go.Box(y=gb["scores_list"], name=agent_b, marker_color="#38BDF8", boxpoints="all", jitter=0.3))
                fig_box.update_layout(
                    title=f"<b>Score Distribution & Volatility Box Plot</b>",
                    yaxis_title="Score ($)",
                    **PLOTLY_LAYOUT_DEFAULTS
                )
                st.plotly_chart(fig_box, use_container_width=True)
