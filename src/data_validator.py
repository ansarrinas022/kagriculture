"""
Data Validation Module for Kaggriculture.
Performs thorough schema checks, missing value diagnostics, duplicate detection,
type validation, and data integrity audits on episode and action records.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataValidator:
    """Validates Kaggriculture episode datasets and generates quality reports."""

    CRITICAL_FIELDS = ["episode_id"]
    RECOMMENDED_FIELDS = ["agent_a", "agent_b", "score_a", "score_b"]

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data validation suite on a DataFrame.
        
        Returns dictionary with:
            - is_valid: bool (whether dataset passes minimum viable criteria)
            - total_rows: int
            - total_columns: int
            - duplicate_rows_count: int
            - missing_by_column: dict
            - issues: list of dicts (severity, message, column)
            - data_types: dict
            - score_stats: dict (summary of score distributions if available)
        """
        issues: List[Dict[str, str]] = []

        if df is None or df.empty:
            return {
                "is_valid": False,
                "total_rows": 0,
                "total_columns": 0,
                "duplicate_rows_count": 0,
                "missing_by_column": {},
                "issues": [{"severity": "ERROR", "message": "Dataset is completely empty or null.", "column": "All"}],
                "data_types": {},
                "score_stats": {}
            }

        total_rows = len(df)
        total_columns = len(df.columns)

        # 1. Critical Column Presence
        for field in cls.CRITICAL_FIELDS:
            if field not in df.columns:
                issues.append({
                    "severity": "ERROR",
                    "message": f"Critical field '{field}' is missing from the dataset. Schema mapping or file verification required.",
                    "column": field
                })

        for field in cls.RECOMMENDED_FIELDS:
            if field not in df.columns:
                issues.append({
                    "severity": "WARNING",
                    "message": f"Recommended field '{field}' is absent. Associated metrics will be computed as unavailable.",
                    "column": field
                })

        # 2. Duplicate Detection
        dup_mask = pd.Series(False, index=df.index)
        if "episode_id" in df.columns:
            dup_mask = df.duplicated(subset=["episode_id"], keep=False)
            dup_count = int(dup_mask.sum())
            if dup_count > 0:
                issues.append({
                    "severity": "WARNING",
                    "message": f"Detected {dup_count} duplicate episode ID records.",
                    "column": "episode_id"
                })
        else:
            dup_count = int(df.duplicated().sum())
            if dup_count > 0:
                issues.append({
                    "severity": "WARNING",
                    "message": f"Detected {dup_count} completely identical rows.",
                    "column": "General"
                })

        # 3. Missing Value Analysis
        missing_by_col = df.isna().sum().to_dict()
        for col, miss_count in missing_by_col.items():
            if miss_count > 0:
                pct = (miss_count / total_rows) * 100
                severity = "ERROR" if col in cls.CRITICAL_FIELDS else ("WARNING" if pct > 20 else "INFO")
                issues.append({
                    "severity": severity,
                    "message": f"Column '{col}' has {miss_count} missing values ({pct:.1f}%).",
                    "column": col
                })

        # 4. Score Sanity Checks
        score_stats = {}
        for sc in ["score_a", "score_b"]:
            if sc in df.columns:
                try:
                    num_s = pd.to_numeric(df[sc], errors="coerce")
                    invalid_scores = num_s.isna() & df[sc].notna()
                    invalid_cnt = int(invalid_scores.sum())
                    if invalid_cnt > 0:
                        issues.append({
                            "severity": "ERROR",
                            "message": f"{invalid_cnt} values in '{sc}' cannot be converted to numeric scores.",
                            "column": sc
                        })
                    valid_s = num_s.dropna()
                    if not valid_s.empty:
                        score_stats[sc] = {
                            "min": float(valid_s.min()),
                            "max": float(valid_s.max()),
                            "mean": float(valid_s.mean()),
                            "std": float(valid_s.std()) if len(valid_s) > 1 else 0.0,
                            "median": float(valid_s.median())
                        }
                except Exception as e:
                    issues.append({
                        "severity": "WARNING",
                        "message": f"Error calculating score statistics for '{sc}': {str(e)}",
                        "column": sc
                    })

        # Determine overall validity
        has_error = any(issue["severity"] == "ERROR" for issue in issues)
        is_valid = (not has_error) and (total_rows > 0)

        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return {
            "is_valid": is_valid,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "duplicate_rows_count": dup_count,
            "missing_by_column": missing_by_col,
            "issues": issues,
            "data_types": data_types,
            "score_stats": score_stats
        }

    @classmethod
    def validate_actions_dataset(cls, actions_df: pd.DataFrame) -> Dict[str, Any]:
        """Validates action sequence records."""
        if actions_df is None or actions_df.empty:
            return {
                "is_valid": False,
                "total_actions": 0,
                "issues": [{"severity": "INFO", "message": "No granular action sequence records present.", "column": "actions"}]
            }

        issues: List[Dict[str, str]] = []
        expected = ["episode_id", "action"]
        for col in expected:
            if col not in actions_df.columns:
                issues.append({
                    "severity": "WARNING",
                    "message": f"Action sequence dataset missing expected column '{col}'",
                    "column": col
                })

        return {
            "is_valid": len([i for i in issues if i["severity"] == "ERROR"]) == 0,
            "total_actions": len(actions_df),
            "unique_episodes": int(actions_df["episode_id"].nunique()) if "episode_id" in actions_df.columns else 0,
            "unique_actions": int(actions_df["action"].nunique()) if "action" in actions_df.columns else 0,
            "issues": issues
        }
