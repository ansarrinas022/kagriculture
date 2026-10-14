"""
Visualization Engine for Kaggriculture.
Generates polished, responsive Plotly interactive visualizations
matching the dark green and emerald agricultural design theme.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import THEME_COLORS


# Custom Plotly Template
PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(19, 42, 34, 0.4)",
    font=dict(color=THEME_COLORS["text_primary"], family="Segoe UI, Roboto, sans-serif"),
    xaxis=dict(
        gridcolor=THEME_COLORS["border_color"],
        zerolinecolor=THEME_COLORS["border_color"],
        showline=True,
        linecolor=THEME_COLORS["border_color"]
    ),
    yaxis=dict(
        gridcolor=THEME_COLORS["border_color"],
        zerolinecolor=THEME_COLORS["border_color"],
        showline=True,
        linecolor=THEME_COLORS["border_color"]
    ),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(11, 25, 20, 0.7)",
        bordercolor=THEME_COLORS["border_color"],
        borderwidth=1
    )
)


class Visualizer:
    """Creates styled Plotly charts for agricultural dashboard modules."""

    @staticmethod
    def plot_score_distribution(episodes_df: pd.DataFrame) -> go.Figure:
        """Plots episode score distribution with histogram and box plot."""
        scores = []
        if "score_a" in episodes_df.columns:
            s_a = pd.to_numeric(episodes_df["score_a"], errors="coerce").dropna()
            scores.extend(s_a.tolist())
        if "score_b" in episodes_df.columns:
            s_b = pd.to_numeric(episodes_df["score_b"], errors="coerce").dropna()
            scores.extend(s_b.tolist())

        if not scores:
            fig = go.Figure()
            fig.update_layout(title="Score Distribution (No Data Available)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=25,
            marker_color=THEME_COLORS["accent_emerald"],
            opacity=0.85,
            name="Score Frequency"
        ))

        fig.update_layout(
            title="<b>Episode Score Distribution</b>",
            xaxis_title="Final Score (Bank Balance $)",
            yaxis_title="Count of Episodes",
            bargap=0.08,
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_agent_leaderboard_bar(leaderboard_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Horizontal bar chart comparing agent average scores with error/std bars."""
        if leaderboard_df.empty:
            fig = go.Figure()
            fig.update_layout(title="Agent Performance (No Data)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        df_sub = leaderboard_df.head(top_n).sort_values(by="avg_score", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_sub["avg_score"],
            y=df_sub["agent"],
            orientation="h",
            marker=dict(
                color=df_sub["avg_score"],
                colorscale=[[0, "#065F46"], [0.5, "#10B981"], [1, "#34D399"]],
                line=dict(color="#10B981", width=1.5)
            ),
            text=[f"${s:,.0f} (Win: {w}%)" for s, w in zip(df_sub["avg_score"], df_sub["win_rate"])],
            textposition="auto"
        ))

        fig.update_layout(
            title=f"<b>Top {min(top_n, len(df_sub))} Agents by Average Final Score</b>",
            xaxis_title="Average Score ($)",
            yaxis_title="",
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_action_distribution(action_counts_df: pd.DataFrame) -> go.Figure:
        """Plots donut chart of action distribution categorized by farming type."""
        if action_counts_df.empty or "count" not in action_counts_df.columns:
            fig = go.Figure()
            fig.update_layout(title="Action Distribution (No Data)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        palette = ["#10B981", "#34D399", "#F59E0B", "#38BDF8", "#A78BFA", "#F472B6", "#EAB308", "#2DD4BF"]

        fig = go.Figure(data=[go.Pie(
            labels=action_counts_df["action"],
            values=action_counts_df["count"],
            hole=0.48,
            marker=dict(colors=palette),
            textinfo="label+percent",
            insidetextorientation="radial"
        )])

        fig.update_layout(
            title="<b>Simulation Farming Action Breakdown</b>",
            showlegend=True,
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_performance_trends(episodes_df: pd.DataFrame) -> go.Figure:
        """Time-series or episode-sequence score trend plot with rolling moving average."""
        if episodes_df.empty or "score_a" not in episodes_df.columns:
            fig = go.Figure()
            fig.update_layout(title="Performance Trends (No Data)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        df_t = episodes_df.copy().reset_index(drop=True)
        df_t["score_a_num"] = pd.to_numeric(df_t["score_a"], errors="coerce")
        df_t["score_b_num"] = pd.to_numeric(df_t.get("score_b"), errors="coerce")
        df_t["avg_score"] = df_t[["score_a_num", "score_b_num"]].mean(axis=1)
        df_t["rolling_avg"] = df_t["avg_score"].rolling(window=min(10, len(df_t)), min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_t.index + 1,
            y=df_t["avg_score"],
            mode="markers",
            marker=dict(color="#10B981", size=6, opacity=0.6),
            name="Episode Score"
        ))
        fig.add_trace(go.Scatter(
            x=df_t.index + 1,
            y=df_t["rolling_avg"],
            mode="lines",
            line=dict(color="#F59E0B", width=3),
            name="10-Match Moving Avg"
        ))

        fig.update_layout(
            title="<b>Performance Pacing Across Episodes</b>",
            xaxis_title="Episode Sequence",
            yaxis_title="Score ($)",
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_outcome_distribution(episodes_df: pd.DataFrame) -> go.Figure:
        """Outcome distribution donut chart (Agent A Win, Agent B Win, Tie)."""
        if episodes_df.empty or "winner" not in episodes_df.columns:
            fig = go.Figure()
            fig.update_layout(title="Outcome Distribution (No Data)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        counts = episodes_df["winner"].value_counts()
        colors = ["#10B981", "#38BDF8", "#F59E0B", "#9CA3AF"]

        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.52,
            marker=dict(colors=colors),
            textinfo="label+percent"
        )])

        fig.update_layout(
            title="<b>Match Victory & Outcome Distribution</b>",
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_agent_radar_comparison(comp_data: Dict[str, Any]) -> go.Figure:
        """Multi-dimensional radar chart comparing strategic farming profiles."""
        ag_a = comp_data.get("agent_a", "Agent A")
        ag_b = comp_data.get("agent_b", "Agent B")
        ga = comp_data.get("global_a", {})
        gb = comp_data.get("global_b", {})

        categories = ["Win Rate (%)", "Avg Harvests", "Avg Sales ($k)", "Median Score ($k)", "Score Stability"]
        
        # Normalize metrics for radar display (0 to 100 scale)
        def scale(val, max_v):
            if val is None or max_v == 0:
                return 10
            return min(100, max(5, float(val) / max_v * 100))

        r_a = [
            ga.get("win_rate", 50),
            scale(ga.get("avg_harvests", 150), 250),
            scale((ga.get("avg_sales", 10000) or 0) / 1000, 20),
            scale((ga.get("median_score", 12000) or 0) / 1000, 25),
            max(10, 100 - min(90, (ga.get("std_score", 1000) or 0) / 40))
        ]

        r_b = [
            gb.get("win_rate", 50),
            scale(gb.get("avg_harvests", 150), 250),
            scale((gb.get("avg_sales", 10000) or 0) / 1000, 20),
            scale((gb.get("median_score", 12000) or 0) / 1000, 25),
            max(10, 100 - min(90, (gb.get("std_score", 1000) or 0) / 40))
        ]

        # Close loop
        categories_closed = categories + [categories[0]]
        r_a_closed = r_a + [r_a[0]]
        r_b_closed = r_b + [r_b[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_a_closed,
            theta=categories_closed,
            fill="toself",
            name=ag_a,
            line=dict(color="#10B981", width=2.5),
            fillcolor="rgba(16, 185, 129, 0.25)"
        ))
        fig.add_trace(go.Scatterpolar(
            r=r_b_closed,
            theta=categories_closed,
            fill="toself",
            name=ag_b,
            line=dict(color="#38BDF8", width=2.5),
            fillcolor="rgba(56, 189, 248, 0.25)"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#9CA3AF", gridcolor=THEME_COLORS["border_color"]),
                angularaxis=dict(color=THEME_COLORS["text_primary"], gridcolor=THEME_COLORS["border_color"])
            ),
            title=f"<b>Head-to-Head Strategy Radar: {ag_a} vs {ag_b}</b>",
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
        """Correlation matrix heatmap with agricultural color palette."""
        if corr_df.empty:
            fig = go.Figure()
            fig.update_layout(title="Correlation Matrix (No Data)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale=[[0.0, "#0F172A"], [0.5, "#065F46"], [1.0, "#34D399"]],
            zmin=-1,
            zmax=1,
            text=np.around(corr_df.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="#ECFDF5", size=12)
        ))

        fig.update_layout(
            title="<b>Numerical Feature Correlation Matrix</b>",
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_replay_timeline(actions_df: pd.DataFrame, episode_id: str) -> go.Figure:
        """Plots step-by-step reward/bank trajectory during an episode."""
        if actions_df.empty or "turn" not in actions_df.columns:
            fig = go.Figure()
            fig.update_layout(title="Replay Timeline (No Granular Data)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        ep_act = actions_df[actions_df["episode_id"] == str(episode_id)].copy()
        if ep_act.empty:
            fig = go.Figure()
            fig.update_layout(title=f"Episode {episode_id} Replay (No Steps Found)", **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        ep_act = ep_act.sort_values(by="turn")

        fig = go.Figure()
        agents = ep_act["agent"].unique()
        colors = ["#10B981", "#38BDF8"]

        for idx, agent in enumerate(agents):
            sub = ep_act[ep_act["agent"] == agent]
            if "reward_after" in sub.columns and sub["reward_after"].notna().any():
                fig.add_trace(go.Scatter(
                    x=sub["turn"],
                    y=sub["reward_after"],
                    mode="lines+markers",
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                    marker=dict(size=5),
                    name=f"{agent} Bank ($)",
                    hovertext=[f"Turn {t}: {a}" for t, a in zip(sub["turn"], sub["action"])]
                ))

        fig.update_layout(
            title=f"<b>Episode {episode_id}: Turn-by-Turn Bank Progression</b>",
            xaxis_title="Turn Index",
            yaxis_title="Bank Balance ($)",
            **PLOTLY_LAYOUT_DEFAULTS
        )
        return fig

    @staticmethod
    def plot_clusters_scatter(cluster_res: Dict[str, Any]) -> go.Figure:
        """2D PCA projection scatter plot for strategy clusters."""
        if "error" in cluster_res:
            fig = go.Figure()
            fig.update_layout(title=cluster_res["error"], **PLOTLY_LAYOUT_DEFAULTS)
            return fig

        df = cluster_res["cluster_df"]
        fig = px.scatter(
            df,
            x="pca_x",
            y="pca_y",
            color="cluster",
            hover_data=["agent", "score", "harvests", "sales", "quadrants"],
            color_continuous_scale=["#10B981", "#F59E0B", "#38BDF8", "#EC4899", "#8B5CF6"],
            title="<b>Farming Strategy Clusters (2D PCA Projection)</b>"
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        return fig
