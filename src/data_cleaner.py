"""
Data Cleaner Module for Kaggriculture.
Applies safe, audit-trailed transformations and cleaning routines
without mutating the original uploaded dataset.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataCleaner:
    """Provides non-destructive cleaning operations and tracks transformations."""

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.current_df = df.copy()
        self.audit_log: List[str] = []

    def get_audit_log(self) -> List[str]:
        return self.audit_log.copy()

    def reset(self) -> pd.DataFrame:
        """Resets to original state."""
        self.current_df = self.original_df.copy()
        self.audit_log = ["Reset dataset to original state."]
        return self.current_df.copy()

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> Tuple[pd.DataFrame, int]:
        """
        Removes duplicate records based on subset (defaults to ['episode_id'] if available).
        """
        initial_count = len(self.current_df)
        cols_to_use = subset
        if not cols_to_use and "episode_id" in self.current_df.columns:
            cols_to_use = ["episode_id"]

        self.current_df = self.current_df.drop_duplicates(subset=cols_to_use, keep=keep)
        removed_count = initial_count - len(self.current_df)
        
        msg = f"Removed {removed_count} duplicate rows (subset: {cols_to_use or 'all columns'}, keep: {keep})."
        self.audit_log.append(msg)
        return self.current_df.copy(), removed_count

    def handle_missing_values(self, num_strategy: str = "median", cat_strategy: str = "Unknown") -> pd.DataFrame:
        """
        Handles missing values using safe strategies.
        num_strategy: 'median', 'mean', 'zero', or 'drop'
        cat_strategy: replacement string or 'drop'
        """
        initial_missing = int(self.current_df.isna().sum().sum())
        if initial_missing == 0:
            self.audit_log.append("Checked missing values: none found.")
            return self.current_df.copy()

        df_cleaned = self.current_df.copy()

        # Numerical columns
        num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_cleaned[col].isna().sum() > 0:
                if num_strategy == "median":
                    fill_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                elif num_strategy == "mean":
                    fill_val = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                elif num_strategy == "zero":
                    df_cleaned[col] = df_cleaned[col].fillna(0.0)

        # Categorical columns
        cat_cols = df_cleaned.select_dtypes(include=["object", "string", "category"]).columns
        for col in cat_cols:
            if df_cleaned[col].isna().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(cat_strategy)

        if num_strategy == "drop" or cat_strategy == "drop":
            df_cleaned = df_cleaned.dropna()

        remaining_missing = int(df_cleaned.isna().sum().sum())
        msg = f"Handled missing values (num: {num_strategy}, cat: {cat_strategy}). Missing count reduced from {initial_missing} to {remaining_missing}."
        self.audit_log.append(msg)
        self.current_df = df_cleaned
        return self.current_df.copy()

    def filter_score_outliers(self, method: str = "iqr", threshold: float = 1.5) -> Tuple[pd.DataFrame, int]:
        """
        Removes episodes with extreme score outliers using IQR or Z-score method.
        """
        initial_count = len(self.current_df)
        score_cols = [c for c in ["score_a", "score_b"] if c in self.current_df.columns]
        if not score_cols:
            self.audit_log.append("Outlier filter skipped: No score columns found.")
            return self.current_df.copy(), 0

        valid_mask = pd.Series(True, index=self.current_df.index)

        for col in score_cols:
            vals = pd.to_numeric(self.current_df[col], errors="coerce")
            if vals.dropna().empty:
                continue

            if method.lower() == "iqr":
                q25 = vals.quantile(0.25)
                q75 = vals.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - (threshold * iqr)
                upper_bound = q75 + (threshold * iqr)
                col_mask = (vals >= lower_bound) & (vals <= upper_bound)
                valid_mask = valid_mask & (col_mask | vals.isna())
            elif method.lower() == "zscore":
                mean = vals.mean()
                std = vals.std()
                if std > 0:
                    z = (vals - mean).abs() / std
                    valid_mask = valid_mask & ((z <= threshold) | vals.isna())

        self.current_df = self.current_df[valid_mask]
        removed_count = initial_count - len(self.current_df)
        msg = f"Filtered {removed_count} outlier episodes using {method.upper()} (threshold: {threshold})."
        self.audit_log.append(msg)
        return self.current_df.copy(), removed_count

    def filter_by_criteria(self, min_turns: Optional[int] = None, agent_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Filters dataset by min turns and agent names."""
        df_f = self.current_df.copy()
        initial_count = len(df_f)

        if min_turns is not None and "total_turns" in df_f.columns:
            df_f = df_f[df_f["total_turns"] >= min_turns]

        if agent_filter and ("agent_a" in df_f.columns or "agent_b" in df_f.columns):
            cond = pd.Series(False, index=df_f.index)
            if "agent_a" in df_f.columns:
                cond = cond | df_f["agent_a"].isin(agent_filter)
            if "agent_b" in df_f.columns:
                cond = cond | df_f["agent_b"].isin(agent_filter)
            df_f = df_f[cond]

        diff = initial_count - len(df_f)
        msg = f"Applied custom criteria filter: {len(df_f)} records retained ({diff} removed)."
        self.audit_log.append(msg)
        self.current_df = df_f
        return self.current_df.copy()
