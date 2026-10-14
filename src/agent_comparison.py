"""
Agent Performance Comparison Module for Kaggriculture.
Performs rigorous head-to-head matched match analysis and global cross-agent comparisons.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class AgentComparison:
    """Compares two competing farming agents across matched and global matches."""

    def __init__(self, episodes_df: pd.DataFrame, actions_df: Optional[pd.DataFrame] = None):
        self.df = episodes_df.copy()
        self.actions_df = actions_df.copy() if actions_df is not None else None

    def get_all_agents(self) -> List[str]:
        """Returns sorted list of all unique agents in the dataset."""
        agents = set()
        if "agent_a" in self.df.columns:
            agents.update(self.df["agent_a"].dropna().unique())
        if "agent_b" in self.df.columns:
            agents.update(self.df["agent_b"].dropna().unique())
        return sorted(list(agents))

    def get_matched_episodes(self, agent_a: str, agent_b: str) -> pd.DataFrame:
        """
        Finds direct head-to-head episodes where agent_a and agent_b faced each other.
        Normalizes agent roles so Agent A is always the first player in comparison view.
        """
        if self.df.empty or "agent_a" not in self.df.columns or "agent_b" not in self.df.columns:
            return pd.DataFrame()

        # Match condition 1: A is agent_a, B is agent_b
        cond1 = (self.df["agent_a"] == agent_a) & (self.df["agent_b"] == agent_b)
        # Match condition 2: B is agent_a, A is agent_b
        cond2 = (self.df["agent_a"] == agent_b) & (self.df["agent_b"] == agent_a)

        df1 = self.df[cond1].copy()
        df1["role_a"] = "agent_a"
        df1["role_b"] = "agent_b"
        df1["score_target_a"] = pd.to_numeric(df1["score_a"], errors="coerce")
        df1["score_target_b"] = pd.to_numeric(df1["score_b"], errors="coerce")
        df1["winner_target"] = df1.get("winner")

        df2 = self.df[cond2].copy()
        df2["role_a"] = "agent_b"
        df2["role_b"] = "agent_a"
        # Invert scores so target_a is always agent_a
        df2["score_target_a"] = pd.to_numeric(df2["score_b"], errors="coerce")
        df2["score_target_b"] = pd.to_numeric(df2["score_a"], errors="coerce")
        df2["winner_target"] = df2.get("winner")

        matched = pd.concat([df1, df2], ignore_index=True)
        if "score_target_a" in matched.columns and "score_target_b" in matched.columns:
            matched["score_diff"] = (matched["score_target_a"] - matched["score_target_b"]).round(2)

        return matched

    def compare_agents(self, agent_a: str, agent_b: str) -> Dict[str, Any]:
        """
        Performs full comparative analysis between agent_a and agent_b.
        Includes direct matched stats and global aggregate stats.
        """
        matched_df = self.get_matched_episodes(agent_a, agent_b)
        matched_count = len(matched_df)

        # Matched Stats
        matched_stats = {}
        if matched_count > 0:
            scores_a = matched_df["score_target_a"].dropna()
            scores_b = matched_df["score_target_b"].dropna()

            wins_a = int((matched_df["winner_target"] == agent_a).sum()) if "winner_target" in matched_df.columns else 0
            wins_b = int((matched_df["winner_target"] == agent_b).sum()) if "winner_target" in matched_df.columns else 0
            ties = matched_count - wins_a - wins_b

            matched_stats = {
                "matched_episodes_count": matched_count,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "win_rate_a": round((wins_a / matched_count) * 100.0, 1),
                "win_rate_b": round((wins_b / matched_count) * 100.0, 1),
                "mean_score_a": round(float(scores_a.mean()), 2) if not scores_a.empty else None,
                "mean_score_b": round(float(scores_b.mean()), 2) if not scores_b.empty else None,
                "median_score_a": round(float(scores_a.median()), 2) if not scores_a.empty else None,
                "median_score_b": round(float(scores_b.median()), 2) if not scores_b.empty else None,
                "max_score_a": round(float(scores_a.max()), 2) if not scores_a.empty else None,
                "max_score_b": round(float(scores_b.max()), 2) if not scores_b.empty else None,
                "min_score_a": round(float(scores_a.min()), 2) if not scores_a.empty else None,
                "min_score_b": round(float(scores_b.min()), 2) if not scores_b.empty else None,
                "std_score_a": round(float(scores_a.std()), 2) if len(scores_a) > 1 else 0.0,
                "std_score_b": round(float(scores_b.std()), 2) if len(scores_b) > 1 else 0.0,
                "avg_score_diff": round(float(matched_df["score_diff"].mean()), 2) if "score_diff" in matched_df.columns else None
            }
        else:
            matched_stats = {
                "matched_episodes_count": 0,
                "note": f"No direct head-to-head episodes found between '{agent_a}' and '{agent_b}' in current dataset."
            }

        # Global Stats (Across all matches played by each agent)
        def get_global_agent_stats(agent_name: str) -> Dict[str, Any]:
            cond_a = self.df["agent_a"] == agent_name if "agent_a" in self.df.columns else pd.Series(False, index=self.df.index)
            cond_b = self.df["agent_b"] == agent_name if "agent_b" in self.df.columns else pd.Series(False, index=self.df.index)

            scores = []
            harvests = []
            sales = []
            wins = 0

            if "score_a" in self.df.columns:
                scores.extend(pd.to_numeric(self.df.loc[cond_a, "score_a"], errors="coerce").dropna().tolist())
                if "crops_harvested_a" in self.df.columns:
                    harvests.extend(pd.to_numeric(self.df.loc[cond_a, "crops_harvested_a"], errors="coerce").dropna().tolist())
                if "market_sales_a" in self.df.columns:
                    sales.extend(pd.to_numeric(self.df.loc[cond_a, "market_sales_a"], errors="coerce").dropna().tolist())

            if "score_b" in self.df.columns:
                scores.extend(pd.to_numeric(self.df.loc[cond_b, "score_b"], errors="coerce").dropna().tolist())
                if "crops_harvested_b" in self.df.columns:
                    harvests.extend(pd.to_numeric(self.df.loc[cond_b, "crops_harvested_b"], errors="coerce").dropna().tolist())
                if "market_sales_b" in self.df.columns:
                    sales.extend(pd.to_numeric(self.df.loc[cond_b, "market_sales_b"], errors="coerce").dropna().tolist())

            if "winner" in self.df.columns:
                wins = int(((cond_a | cond_b) & (self.df["winner"] == agent_name)).sum())

            total_games = int((cond_a | cond_b).sum())

            s_series = pd.Series(scores)
            iqr = round(float(s_series.quantile(0.75) - s_series.quantile(0.25)), 2) if len(s_series) > 3 else 0.0

            return {
                "total_episodes": total_games,
                "total_wins": wins,
                "win_rate": round((wins / total_games) * 100.0, 1) if total_games > 0 else 0.0,
                "mean_score": round(float(s_series.mean()), 2) if not s_series.empty else None,
                "median_score": round(float(s_series.median()), 2) if not s_series.empty else None,
                "max_score": round(float(s_series.max()), 2) if not s_series.empty else None,
                "min_score": round(float(s_series.min()), 2) if not s_series.empty else None,
                "std_score": round(float(s_series.std()), 2) if len(s_series) > 1 else 0.0,
                "iqr_score": iqr,
                "avg_harvests": round(float(np.mean(harvests)), 1) if harvests else None,
                "avg_sales": round(float(np.mean(sales)), 2) if sales else None,
                "scores_list": scores
            }

        global_a = get_global_agent_stats(agent_a)
        global_b = get_global_agent_stats(agent_b)

        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "matched_stats": matched_stats,
            "matched_df": matched_df,
            "global_a": global_a,
            "global_b": global_b
        }
