"""
Episode Analyzer Module for Kaggriculture.
Calculates global and filtered metrics, agent rankings, score distributions,
and match performance across episodes.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class EpisodeAnalyzer:
    """Analyzes simulation episodes, scores, agents, and game dynamics."""

    def __init__(self, episodes_df: pd.DataFrame):
        self.df = episodes_df.copy()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Computes high-level summary metrics from current dataset.
        Distinguishes missing fields from zero values.
        """
        if self.df.empty:
            return {
                "total_episodes": 0,
                "unique_agents_count": 0,
                "total_actions": None,
                "avg_score": None,
                "max_score": None,
                "min_score": None,
                "win_rate_available": False,
                "valid_records_count": 0,
                "dataset_size_bytes": 0
            }

        total_episodes = len(self.df)

        # Unique agents
        agents = set()
        if "agent_a" in self.df.columns:
            agents.update(self.df["agent_a"].dropna().unique())
        if "agent_b" in self.df.columns:
            agents.update(self.df["agent_b"].dropna().unique())
        unique_agents_count = len(agents)

        # Combined scores from score_a and score_b
        score_vals = []
        if "score_a" in self.df.columns:
            s_a = pd.to_numeric(self.df["score_a"], errors="coerce").dropna()
            score_vals.extend(s_a.tolist())
        if "score_b" in self.df.columns:
            s_b = pd.to_numeric(self.df["score_b"], errors="coerce").dropna()
            score_vals.extend(s_b.tolist())

        avg_score = round(float(np.mean(score_vals)), 2) if score_vals else None
        max_score = round(float(np.max(score_vals)), 2) if score_vals else None
        min_score = round(float(np.min(score_vals)), 2) if score_vals else None

        # Total actions recorded
        total_actions = None
        act_cols = [c for c in ["actions_a_count", "actions_b_count"] if c in self.df.columns]
        if act_cols:
            total_actions = int(self.df[act_cols].sum().sum())

        # Win rate availability
        has_winner = "winner" in self.df.columns and not self.df["winner"].isna().all()

        return {
            "total_episodes": total_episodes,
            "unique_agents_count": unique_agents_count,
            "unique_agents_list": sorted(list(agents)),
            "total_actions": total_actions,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "win_rate_available": has_winner,
            "valid_records_count": len(self.df.dropna(subset=["episode_id"])) if "episode_id" in self.df.columns else total_episodes,
            "dataset_size_bytes": int(self.df.memory_usage(deep=True).sum())
        }

    def get_agent_leaderboard(self) -> pd.DataFrame:
        """
        Computes comprehensive agent performance leaderboard across all episodes.
        """
        if self.df.empty or "agent_a" not in self.df.columns or "score_a" not in self.df.columns:
            return pd.DataFrame()

        records = []
        # Process Agent A occurrences
        df_a = self.df.dropna(subset=["agent_a"]).copy()
        df_a["score"] = pd.to_numeric(df_a["score_a"], errors="coerce")
        df_a["is_win"] = (df_a["winner"] == df_a["agent_a"]) if "winner" in df_a.columns else False
        df_a["agent"] = df_a["agent_a"]
        df_a["harvested"] = pd.to_numeric(df_a.get("crops_harvested_a", 0), errors="coerce").fillna(0)
        df_a["sales"] = pd.to_numeric(df_a.get("market_sales_a", 0), errors="coerce").fillna(0)

        # Process Agent B occurrences
        df_b = self.df.dropna(subset=["agent_b"]).copy()
        df_b["score"] = pd.to_numeric(df_b["score_b"], errors="coerce")
        df_b["is_win"] = (df_b["winner"] == df_b["agent_b"]) if "winner" in df_b.columns else False
        df_b["agent"] = df_b["agent_b"]
        df_b["harvested"] = pd.to_numeric(df_b.get("crops_harvested_b", 0), errors="coerce").fillna(0)
        df_b["sales"] = pd.to_numeric(df_b.get("market_sales_b", 0), errors="coerce").fillna(0)

        combined = pd.concat([
            df_a[["agent", "score", "is_win", "harvested", "sales"]],
            df_b[["agent", "score", "is_win", "harvested", "sales"]]
        ], ignore_index=True)

        if combined.empty:
            return pd.DataFrame()

        grouped = combined.groupby("agent").agg(
            episodes=("score", "count"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            max_score=("score", "max"),
            min_score=("score", "min"),
            score_std=("score", "std"),
            total_wins=("is_win", "sum"),
            avg_harvested=("harvested", "mean"),
            avg_market_sales=("sales", "mean")
        ).reset_index()

        grouped["win_rate"] = (grouped["total_wins"] / grouped["episodes"]) * 100.0
        grouped["win_rate"] = grouped["win_rate"].round(1)
        grouped["avg_score"] = grouped["avg_score"].round(2)
        grouped["median_score"] = grouped["median_score"].round(2)
        grouped["score_std"] = grouped["score_std"].fillna(0).round(2)
        grouped["avg_harvested"] = grouped["avg_harvested"].round(1)
        grouped["avg_market_sales"] = grouped["avg_market_sales"].round(2)

        return grouped.sort_values(by="avg_score", ascending=False).reset_index(drop=True)

    def filter_episodes(self, 
                        agent: Optional[str] = None, 
                        min_score: Optional[float] = None, 
                        max_score: Optional[float] = None,
                        winner: Optional[str] = None,
                        search_id: Optional[str] = None) -> pd.DataFrame:
        """Filters episodes based on interactive UI selections."""
        res = self.df.copy()

        if search_id and search_id.strip() and "episode_id" in res.columns:
            res = res[res["episode_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]

        if agent and agent != "All Agents":
            cond = pd.Series(False, index=res.index)
            if "agent_a" in res.columns:
                cond = cond | (res["agent_a"] == agent)
            if "agent_b" in res.columns:
                cond = cond | (res["agent_b"] == agent)
            res = res[cond]

        if min_score is not None:
            cond = pd.Series(True, index=res.index)
            if "score_a" in res.columns and "score_b" in res.columns:
                cond = (pd.to_numeric(res["score_a"], errors="coerce") >= min_score) | \
                       (pd.to_numeric(res["score_b"], errors="coerce") >= min_score)
            res = res[cond]

        if max_score is not None:
            cond = pd.Series(True, index=res.index)
            if "score_a" in res.columns and "score_b" in res.columns:
                cond = (pd.to_numeric(res["score_a"], errors="coerce") <= max_score) & \
                       (pd.to_numeric(res["score_b"], errors="coerce") <= max_score)
            res = res[cond]

        if winner and winner != "All Outcomes" and "winner" in res.columns:
            res = res[res["winner"] == winner]

        return res
