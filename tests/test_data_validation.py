"""
Unit tests for DataValidator and DataCleaner.
"""
import pytest
import pandas as pd
from src.data_validator import DataValidator
from src.data_cleaner import DataCleaner


def test_validator_with_valid_data():
    df = pd.DataFrame({
        "episode_id": ["ep_1", "ep_2", "ep_3"],
        "agent_a": ["Alpha", "Beta", "Alpha"],
        "agent_b": ["Beta", "Alpha", "Beta"],
        "score_a": [1000.0, 1500.0, 1200.0],
        "score_b": [900.0, 1600.0, 1100.0]
    })
    res = DataValidator.validate_dataset(df)
    assert res["is_valid"] is True
    assert res["total_rows"] == 3
    assert res["duplicate_rows_count"] == 0


def test_validator_detects_duplicates_and_missing():
    df = pd.DataFrame({
        "episode_id": ["ep_1", "ep_1", None],
        "score_a": [100.0, None, 300.0]
    })
    res = DataValidator.validate_dataset(df)
    assert res["duplicate_rows_count"] >= 1
    assert res["missing_by_column"]["episode_id"] == 1


def test_data_cleaner_deduplication():
    df = pd.DataFrame({
        "episode_id": ["ep_1", "ep_1", "ep_2"],
        "score_a": [100, 100, 200]
    })
    cleaner = DataCleaner(df)
    cleaned_df, removed = cleaner.remove_duplicates()
    assert removed == 1
    assert len(cleaned_df) == 2
    assert len(cleaner.get_audit_log()) > 0
