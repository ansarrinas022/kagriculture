"""
State Manager Module for Kaggriculture.
Handles Streamlit session state initialization, global filters, and caching.
"""
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import pandas as pd

from config import SAMPLE_CSV_PATH, SAMPLE_JSON_PATH
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.episode_analyzer import EpisodeAnalyzer
from src.action_analyzer import ActionAnalyzer


def initialize_session_state():
    """Initializes global session state variables if not already present."""
    if "episodes_df" not in st.session_state:
        # Load default sample dataset so application starts with immediately working analytics
        try:
            if SAMPLE_CSV_PATH.exists():
                ep_df, _, meta = DataLoader.load_from_file(SAMPLE_CSV_PATH)
                st.session_state["episodes_df"] = ep_df
                st.session_state["dataset_metadata"] = {
                    "source": "Sample Benchmark Dataset",
                    "filename": SAMPLE_CSV_PATH.name,
                    "is_sample": True
                }
            else:
                ep_df, act_df = DataLoader.generate_sample_dataset()
                st.session_state["episodes_df"] = ep_df
                st.session_state["actions_df"] = act_df
                st.session_state["dataset_metadata"] = {
                    "source": "Generated Sample Dataset",
                    "filename": "sample_episodes.csv",
                    "is_sample": True
                }
        except Exception:
            st.session_state["episodes_df"] = pd.DataFrame()
            st.session_state["dataset_metadata"] = {"source": "None", "filename": "None", "is_sample": False}

    if "actions_df" not in st.session_state:
        try:
            if SAMPLE_JSON_PATH.exists():
                _, act_df, _ = DataLoader.load_from_file(SAMPLE_JSON_PATH)
                st.session_state["actions_df"] = act_df
            else:
                st.session_state["actions_df"] = None
        except Exception:
            st.session_state["actions_df"] = None

    if "validation_report" not in st.session_state:
        if not st.session_state["episodes_df"].empty:
            st.session_state["validation_report"] = DataValidator.validate_dataset(st.session_state["episodes_df"])
        else:
            st.session_state["validation_report"] = {}

    if "cleaner_audit_log" not in st.session_state:
        st.session_state["cleaner_audit_log"] = []

    if "active_filters" not in st.session_state:
        st.session_state["active_filters"] = {
            "selected_agent": "All Agents",
            "selected_outcome": "All Outcomes",
            "score_range": None,
            "search_query": ""
        }


def render_sidebar_status_and_filters() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Renders sidebar dataset status, badge, and interactive global filters.
    Returns:
        filtered_episodes_df: pd.DataFrame
        filtered_actions_df: Optional[pd.DataFrame]
    """
    initialize_session_state()

    df = st.session_state["episodes_df"]
    actions_df = st.session_state.get("actions_df")
    meta = st.session_state.get("dataset_metadata", {})

    with st.sidebar:
        st.markdown("### 🌱 **Kaggriculture**")
        st.markdown("<span style='font-size:0.8rem; color:#9CA3AF;'>Simulation Analysis & Strategy</span>", unsafe_allow_html=True)
        st.markdown("---")

        # Dataset Status Indicator
        if meta.get("is_sample", False):
            st.markdown(
                """<div style='background:rgba(245, 158, 11, 0.15); border:1px solid #F59E0B; padding:8px 12px; border-radius:8px; margin-bottom:12px;'>
                <span style='color:#FBBF24; font-weight:600; font-size:0.85rem;'>⚠️ Active: Sample Dataset</span><br>
                <span style='color:#9CA3AF; font-size:0.75rem;'>Use Data Management to upload official Kaggle simulation files.</span>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style='background:rgba(16, 185, 129, 0.15); border:1px solid #10B981; padding:8px 12px; border-radius:8px; margin-bottom:12px;'>
                <span style='color:#34D399; font-weight:600; font-size:0.85rem;'>✅ Active: {meta.get('filename', 'Custom Dataset')}</span><br>
                <span style='color:#9CA3AF; font-size:0.75rem;'>Records: {len(df):,} episodes</span>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("#### 🔍 **Global Filters**")

        # 1. Agent Filter
        agents = ["All Agents"]
        if not df.empty:
            analyzer = EpisodeAnalyzer(df)
            agents.extend(analyzer.get_summary_metrics().get("unique_agents_list", []))

        selected_agent = st.selectbox(
            "Agent Filter",
            options=agents,
            index=0,
            help="Filter analysis to episodes involving a specific agent."
        )

        # 2. Outcome Filter
        outcome_options = ["All Outcomes"]
        if not df.empty and "winner" in df.columns:
            outcomes = [w for w in df["winner"].dropna().unique() if w]
            outcome_options.extend(outcomes)

        selected_outcome = st.selectbox(
            "Outcome Filter",
            options=outcome_options,
            index=0,
            help="Filter by episode winner or match result."
        )

        # 3. Score Range Filter
        min_score = 0.0
        max_score = 25000.0
        if not df.empty and "score_a" in df.columns:
            s_num = pd.to_numeric(df["score_a"], errors="coerce").dropna()
            if not s_num.empty:
                min_score = float(max(0.0, s_num.min()))
                max_score = float(s_num.max() + 500)

        score_range = st.slider(
            "Final Score Range ($)",
            min_value=float(min_score),
            max_value=float(max(max_score, min_score + 100)),
            value=(float(min_score), float(max_score)),
            step=250.0,
            help="Filter episodes where agent score falls within specified range."
        )

        st.session_state["active_filters"] = {
            "selected_agent": selected_agent,
            "selected_outcome": selected_outcome,
            "score_range": score_range
        }

        # Quick Actions
        st.markdown("---")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state["active_filters"] = {
                    "selected_agent": "All Agents",
                    "selected_outcome": "All Outcomes",
                    "score_range": (float(min_score), float(max_score))
                }
                st.rerun()

    # Apply filters
    filtered_df = df.copy()
    if not filtered_df.empty:
        analyzer = EpisodeAnalyzer(filtered_df)
        filtered_df = analyzer.filter_episodes(
            agent=selected_agent if selected_agent != "All Agents" else None,
            min_score=score_range[0],
            max_score=score_range[1],
            winner=selected_outcome if selected_outcome != "All Outcomes" else None
        )

    # Filter actions dataset if present
    filtered_actions = None
    if actions_df is not None and not actions_df.empty and not filtered_df.empty:
        valid_episodes = set(filtered_df["episode_id"].astype(str).unique())
        filtered_actions = actions_df[actions_df["episode_id"].astype(str).isin(valid_episodes)].copy()
        if selected_agent != "All Agents":
            filtered_actions = filtered_actions[filtered_actions["agent"] == selected_agent]

    return filtered_df, filtered_actions
