"""
Statistical Analysis Module for Kaggriculture.
Computes descriptive statistics, distributions, correlations, and outlier diagnostics
with mathematical safeguards against division-by-zero, constant variables, and small samples.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class StatisticalAnalysis:
    """Provides statistical evaluations and distribution analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_numerical_columns(self) -> List[str]:
        """Returns list of numeric columns with at least some non-null values."""
        num_cols = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                num_cols.append(col)
            else:
                # Check if coercible
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() > 3:
                    num_cols.append(col)
        return sorted(num_cols)

    def compute_descriptive_stats(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Computes detailed descriptive statistics table for selected numerical columns.
        Includes mean, std, min, 25%, median, 75%, IQR, max, skewness, missing count.
        """
        target_cols = columns or self.get_numerical_columns()
        if not target_cols:
            return pd.DataFrame()

        rows = []
        for col in target_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            valid = s.dropna()
            n_valid = len(valid)

            if n_valid == 0:
                rows.append({
                    "Feature": col,
                    "Count": 0,
                    "Missing": len(self.df),
                    "Missing %": 100.0,
                    "Mean": None,
                    "Std Dev": None,
                    "Min": None,
                    "Q1 (25%)": None,
                    "Median": None,
                    "Q3 (75%)": None,
                    "IQR": None,
                    "Max": None,
                    "Skewness": None
                })
                continue

            q25 = float(valid.quantile(0.25))
            q75 = float(valid.quantile(0.75))
            iqr = q75 - q25

            # Skewness requires at least 3 points
            skew_val = round(float(valid.skew()), 3) if n_valid >= 3 else 0.0

            rows.append({
                "Feature": col,
                "Count": n_valid,
                "Missing": int(s.isna().sum()),
                "Missing %": round((s.isna().sum() / len(self.df)) * 100, 1),
                "Mean": round(float(valid.mean()), 2),
                "Std Dev": round(float(valid.std()), 2) if n_valid > 1 else 0.0,
                "Min": round(float(valid.min()), 2),
                "Q1 (25%)": round(q25, 2),
                "Median": round(float(valid.median()), 2),
                "Q3 (75%)": round(q75, 2),
                "IQR": round(iqr, 2),
                "Max": round(float(valid.max()), 2),
                "Skewness": skew_val
            })

        return pd.DataFrame(rows)

    def compute_correlation_matrix(self, columns: Optional[List[str]] = None, method: str = "pearson") -> Tuple[pd.DataFrame, str]:
        """
        Computes pairwise correlation matrix for numerical features.
        Safeguards against zero-variance/constant features.
        """
        cols = columns or self.get_numerical_columns()
        if len(cols) < 2:
            return pd.DataFrame(), "At least two numerical columns are required for correlation analysis."

        num_df = self.df[cols].apply(pd.to_numeric, errors="coerce")
        # Filter out zero variance columns
        valid_cols = [c for c in cols if num_df[c].std() > 1e-9]

        if len(valid_cols) < 2:
            return pd.DataFrame(), "Insufficient variance in selected columns to compute correlations."

        corr_matrix = num_df[valid_cols].corr(method=method).round(3)
        caution_note = "Note: Statistical correlation indicates mathematical association and does NOT imply causation."
        return corr_matrix, caution_note

    def detect_outliers(self, column: str, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
        """
        Identifies outlier records for a given feature using IQR or Z-Score.
        """
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found."}

        vals = pd.to_numeric(self.df[column], errors="coerce")
        valid = vals.dropna()

        if len(valid) < 5:
            return {"error": "Insufficient sample size (minimum 5 valid observations required)."}

        outlier_indices = []
        bounds = {}

        if method.lower() == "iqr":
            q25 = valid.quantile(0.25)
            q75 = valid.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (threshold * iqr)
            upper_bound = q75 + (threshold * iqr)
            bounds = {"lower": round(float(lower_bound), 2), "upper": round(float(upper_bound), 2), "iqr": round(float(iqr), 2)}
            mask = (vals < lower_bound) | (vals > upper_bound)
            outlier_indices = self.df[mask].index.tolist()

        elif method.lower() == "zscore":
            mean = valid.mean()
            std = valid.std()
            if std == 0:
                return {"error": "Standard deviation is zero; cannot compute Z-scores."}
            z_scores = (vals - mean).abs() / std
            bounds = {"mean": round(float(mean), 2), "std": round(float(std), 2), "z_thresh": threshold}
            mask = z_scores > threshold
            outlier_indices = self.df[mask].index.tolist()

        outlier_df = self.df.loc[outlier_indices].copy()
        
        return {
            "column": column,
            "method": method.upper(),
            "threshold": threshold,
            "bounds": bounds,
            "total_outliers": len(outlier_indices),
            "outlier_percentage": round((len(outlier_indices) / len(self.df)) * 100.0, 2),
            "outlier_rows": outlier_df
        }

    def compute_grouped_statistics(self, group_by_col: str, num_col: str) -> pd.DataFrame:
        """
        Computes grouped aggregations dynamically.
        """
        if group_by_col not in self.df.columns or num_col not in self.df.columns:
            return pd.DataFrame()

        clean = self.df[[group_by_col, num_col]].copy()
        clean[num_col] = pd.to_numeric(clean[num_col], errors="coerce")
        clean = clean.dropna()

        if clean.empty:
            return pd.DataFrame()

        grouped = clean.groupby(group_by_col)[num_col].agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        ).reset_index()

        grouped["mean"] = grouped["mean"].round(2)
        grouped["median"] = grouped["median"].round(2)
        grouped["std"] = grouped["std"].fillna(0).round(2)
        return grouped.sort_values(by="mean", ascending=False)
