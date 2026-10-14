"""
Page 2: Episode Explorer
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd
import json

from config import APP_TITLE, CUSTOM_CSS
from src.state_manager import render_sidebar_status_and_filters
from src.action_analyzer import ActionAnalyzer
from src.visualization import Visualizer

st.set_page_config(page_title=f"Episode Explorer | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("🔍 Episode Explorer & Replay Inspector")
st.markdown("<p style='color:#9CA3AF;'>Browse individual simulation matches, inspect chronologically ordered action logs, and export match records.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No episodes match the selected global filters. Reset filters in the sidebar to view episodes.")
else:
    # Local Search and Sorting Controls
    col_search, col_sort, col_order = st.columns([2, 1, 1])
    with col_search:
        search_id = st.text_input("🔎 Search by Episode ID", placeholder="e.g. ep_0012 or match hash...")
    with col_sort:
        sort_options = [c for c in ["score_a", "score_b", "episode_id", "crops_harvested_a", "total_turns"] if c in filtered_df.columns]
        sort_by = st.selectbox("Sort By", options=sort_options, index=0 if "score_a" in sort_options else 0)
    with col_order:
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

    df_display = filtered_df.copy()
    if search_id.strip() and "episode_id" in df_display.columns:
        df_display = df_display[df_display["episode_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]

    if sort_by in df_display.columns:
        ascending = (sort_order == "Ascending")
        if pd.api.types.is_numeric_dtype(df_display[sort_by]):
            df_display = df_display.sort_values(by=sort_by, ascending=ascending)
        else:
            try:
                df_display["_sort_num"] = pd.to_numeric(df_display[sort_by], errors="coerce")
                df_display = df_display.sort_values(by="_sort_num", ascending=ascending).drop(columns=["_sort_num"])
            except Exception:
                df_display = df_display.sort_values(by=sort_by, ascending=ascending)

    # Pagination
    PAGE_SIZE = 15
    total_matches = len(df_display)
    total_pages = max(1, (total_matches + PAGE_SIZE - 1) // PAGE_SIZE)
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.markdown(f"<span style='color:#9CA3AF; font-size:0.9rem;'>Found <b>{total_matches:,}</b> matches matching criteria. Showing page:</span>", unsafe_allow_html=True)
    with col_p2:
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page_num - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_matches)
    page_df = df_display.iloc[start_idx:end_idx]

    # Episodes Table
    st.dataframe(
        page_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 🔬 **Detailed Episode Inspector**")

    # Select episode to inspect
    available_episodes = df_display["episode_id"].astype(str).tolist() if "episode_id" in df_display.columns else []
    if available_episodes:
        selected_ep_id = st.selectbox("Select Episode ID to Inspect", options=available_episodes, index=0)
        ep_row = df_display[df_display["episode_id"].astype(str) == selected_ep_id].iloc[0]

        # Episode Detail Cards
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.markdown(f"""
            <div class="kag-card">
                <span class="kag-badge">Episode Identity</span>
                <h3 style="color:#10B981; margin:8px 0 4px 0;">{selected_ep_id}</h3>
                <span style="color:#9CA3AF; font-size:0.85rem;">Status: {ep_row.get('status', 'FINISHED')}</span>
            </div>
            """, unsafe_allow_html=True)

        with d_col2:
            st.markdown(f"""
            <div class="kag-card">
                <span class="kag-badge">Agent A Performance</span>
                <h3 style="color:#34D399; margin:8px 0 4px 0;">{ep_row.get('agent_a', 'Agent A')}</h3>
                <span style="color:#ECFDF5; font-size:0.95rem;">Score: <b>${float(ep_row.get('score_a', 0)):,.2f}</b></span><br>
                <span style="color:#9CA3AF; font-size:0.8rem;">Harvests: {ep_row.get('crops_harvested_a', 'N/A')} | Quadrants: {ep_row.get('land_quadrants_a', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

        with d_col3:
            st.markdown(f"""
            <div class="kag-card">
                <span class="kag-badge-info">Agent B Performance</span>
                <h3 style="color:#38BDF8; margin:8px 0 4px 0;">{ep_row.get('agent_b', 'Agent B')}</h3>
                <span style="color:#ECFDF5; font-size:0.95rem;">Score: <b>${float(ep_row.get('score_b', 0)):,.2f}</b></span><br>
                <span style="color:#9CA3AF; font-size:0.8rem;">Harvests: {ep_row.get('crops_harvested_b', 'N/A')} | Quadrants: {ep_row.get('land_quadrants_b', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

        with d_col4:
            winner = ep_row.get('winner', 'Undetermined')
            st.markdown(f"""
            <div class="kag-card">
                <span class="kag-badge-gold">Match Outcome</span>
                <h3 style="color:#FBBF24; margin:8px 0 4px 0;">🏆 {winner}</h3>
                <span style="color:#9CA3AF; font-size:0.85rem;">Simulation Days: {ep_row.get('season_day', 30)} (Turns: {ep_row.get('total_turns', 720)})</span>
            </div>
            """, unsafe_allow_html=True)

        # Replay Sequence & Actions Timeline
        action_analyzer = ActionAnalyzer(filtered_df, filtered_actions_df)
        if action_analyzer.has_granular_actions():
            st.markdown("#### ⏱️ **Turn-by-Turn Replay Progression**")
            st.plotly_chart(Visualizer.plot_replay_timeline(filtered_actions_df, selected_ep_id), use_container_width=True)

            chrono_actions = action_analyzer.get_chronological_actions(selected_ep_id)
            if not chrono_actions.empty:
                st.markdown("#### 📜 **Chronological Action Stream**")
                st.dataframe(
                    chrono_actions,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(f"No granular action replay records recorded for episode {selected_ep_id}.")
        else:
            st.info("ℹ️ Turn-by-turn replay actions are not present for this file. Granular action streams require simulation replay JSON or JSONL shards. High-level episode metadata is fully preserved.")

        # Export Episode Data
        st.markdown("#### 💾 **Export Episode Record**")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            ep_csv = ep_row.to_frame().T.to_csv(index=False)
            st.download_button(
                label="📥 Export Episode CSV",
                data=ep_csv,
                file_name=f"kaggriculture_{selected_ep_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with e_col2:
            ep_json = json.dumps(ep_row.to_dict(), indent=2, default=str)
            st.download_button(
                label="📥 Export Episode JSON",
                data=ep_json,
                file_name=f"kaggriculture_{selected_ep_id}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("No episode records available to inspect.")
