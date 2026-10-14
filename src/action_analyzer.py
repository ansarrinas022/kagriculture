"""
Farming Action Analyzer for Kaggriculture.
Analyzes farming activities (planting, watering, harvesting, livestock, trading),
action sequences, temporal patterns, and correlations with victory/scores.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from config import ACTION_CATEGORIES, KNOWN_ACTION_TYPES


class ActionAnalyzer:
    """Analyzes granular action events and macro-level action summaries."""

    def __init__(self, episodes_df: pd.DataFrame, actions_df: Optional[pd.DataFrame] = None):
        self.episodes_df = episodes_df.copy()
        self.actions_df = actions_df.copy() if actions_df is not None else None

    def has_granular_actions(self) -> bool:
        return self.actions_df is not None and not self.actions_df.empty

    def get_action_frequencies(self) -> pd.DataFrame:
        """
        Returns frequencies of all recorded actions.
        Works with granular action logs if available, or derives summary from episodes_df.
        """
        if self.has_granular_actions() and "action" in self.actions_df.columns:
            counts = self.actions_df["action"].value_counts().reset_index()
            counts.columns = ["action", "count"]
            total = counts["count"].sum()
            counts["percentage"] = ((counts["count"] / total) * 100.0).round(2)
            
            # Categorize actions
            def map_cat(act):
                for cat, act_list in ACTION_CATEGORIES.items():
                    if act in act_list or any(a in act for a in act_list):
                        return cat
                return "Other Operations"

            counts["category"] = counts["action"].apply(map_cat)
            return counts.sort_values(by="count", ascending=False)

        # Fallback to episode macro features
        macro_counts = []
        if "crops_harvested_a" in self.episodes_df.columns and "crops_harvested_b" in self.episodes_df.columns:
            total_harvest = int(self.episodes_df["crops_harvested_a"].sum() + self.episodes_df["crops_harvested_b"].sum())
            macro_counts.append({"action": "HARVEST_CROPS", "count": total_harvest, "category": "Crop Operations"})

        if "market_sales_a" in self.episodes_df.columns and "market_sales_b" in self.episodes_df.columns:
            # Approximate transactions
            macro_counts.append({"action": "SELL_MARKET", "count": int(len(self.episodes_df) * 45), "category": "Commerce & Trading"})

        if "land_quadrants_a" in self.episodes_df.columns and "land_quadrants_b" in self.episodes_df.columns:
            quad_buys = int(self.episodes_df["land_quadrants_a"].sum() + self.episodes_df["land_quadrants_b"].sum())
            macro_counts.append({"action": "BUY_QUADRANT", "count": quad_buys, "category": "Farm Expansion"})

        if macro_counts:
            df_m = pd.DataFrame(macro_counts)
            tot = df_m["count"].sum()
            df_m["percentage"] = ((df_m["count"] / tot) * 100.0).round(2)
            return df_m

        return pd.DataFrame(columns=["action", "count", "percentage", "category"])

    def get_actions_by_agent(self) -> pd.DataFrame:
        """
        Returns action distribution segmented by agent.
        """
        if self.has_granular_actions() and "agent" in self.actions_df.columns and "action" in self.actions_df.columns:
            cross = pd.crosstab(self.actions_df["agent"], self.actions_df["action"])
            return cross

        # Fallback using episode macro metrics
        if "agent_a" in self.episodes_df.columns and "actions_a_count" in self.episodes_df.columns:
            df_a = self.episodes_df[["agent_a", "actions_a_count"]].rename(columns={"agent_a": "agent", "actions_a_count": "total_actions"})
            df_b = self.episodes_df[["agent_b", "actions_b_count"]].rename(columns={"agent_b": "agent", "actions_b_count": "total_actions"})
            comb = pd.concat([df_a, df_b]).groupby("agent")["total_actions"].sum().reset_index()
            return comb

        return pd.DataFrame()

    def get_chronological_actions(self, episode_id: str) -> pd.DataFrame:
        """
        Returns chronological action sequence for a specific episode.
        """
        if not self.has_granular_actions():
            return pd.DataFrame()

        ep_actions = self.actions_df[self.actions_df["episode_id"] == str(episode_id)].copy()
        if ep_actions.empty:
            return pd.DataFrame()

        if "turn" in ep_actions.columns:
            ep_actions = ep_actions.sort_values(by="turn", ascending=True)

        return ep_actions.reset_index(drop=True)

    def compare_winning_vs_losing_actions(self) -> pd.DataFrame:
        """
        Compares action profile between winning agents and losing agents.
        """
        if not self.has_granular_actions() or "winner" not in self.episodes_df.columns:
            # Compute based on episode macro aggregates
            if "winner" in self.episodes_df.columns and "crops_harvested_a" in self.episodes_df.columns:
                records = []
                for _, row in self.episodes_df.iterrows():
                    win = row.get("winner")
                    # Agent A
                    is_win_a = (win == row.get("agent_a"))
                    records.append({
                        "outcome": "Winning Episodes" if is_win_a else "Losing Episodes",
                        "avg_harvests": row.get("crops_harvested_a", 0),
                        "avg_sales": row.get("market_sales_a", 0),
                        "quadrants": row.get("land_quadrants_a", 1),
                        "livestock": row.get("livestock_count_a", 0)
                    })
                    # Agent B
                    is_win_b = (win == row.get("agent_b"))
                    records.append({
                        "outcome": "Winning Episodes" if is_win_b else "Losing Episodes",
                        "avg_harvests": row.get("crops_harvested_b", 0),
                        "avg_sales": row.get("market_sales_b", 0),
                        "quadrants": row.get("land_quadrants_b", 1),
                        "livestock": row.get("livestock_count_b", 0)
                    })
                df_res = pd.DataFrame(records).groupby("outcome").mean().round(2).reset_index()
                return df_res
            return pd.DataFrame()

        # Merge actions with episode winners
        merged = self.actions_df.merge(self.episodes_df[["episode_id", "winner"]], on="episode_id", how="inner")
        merged["is_winner"] = merged["agent"] == merged["winner"]

        win_counts = merged[merged["is_winner"]]["action"].value_counts(normalize=True) * 100
        loss_counts = merged[~merged["is_winner"]]["action"].value_counts(normalize=True) * 100

        comp_df = pd.DataFrame({
            "Winner Action %": win_counts,
            "Losing Action %": loss_counts
        }).fillna(0)
        comp_df["Difference (% pts)"] = (comp_df["Winner Action %"] - comp_df["Losing Action %"]).round(2)
        comp_df = comp_df.round(2).reset_index().rename(columns={"index": "action"})
        return comp_df.sort_values(by="Difference (% pts)", ascending=False)
