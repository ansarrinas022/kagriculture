"""
Unit tests for DataLoader and SchemaMapper.
"""
import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoader
from src.schema_mapper import SchemaMapper
from config import SAMPLE_CSV_PATH


def test_schema_mapper_aliases():
    raw_df = pd.DataFrame({
        "EpisodeId": ["ep_001", "ep_002"],
        "Player_1": ["BotA", "BotC"],
        "Player_2": ["BotB", "BotD"],
        "Bank_1": [12000.5, 9500.0],
        "Bank_2": [11500.0, 10200.0]
    })
    mapped = SchemaMapper.apply_mapping(raw_df)
    assert "episode_id" in mapped.columns
    assert "agent_a" in mapped.columns
    assert "agent_b" in mapped.columns
    assert "score_a" in mapped.columns
    assert "score_b" in mapped.columns
    assert "winner" in mapped.columns
    assert mapped["winner"].iloc[0] == "BotA"
    assert mapped["winner"].iloc[1] == "BotD"


def test_load_sample_csv():
    assert SAMPLE_CSV_PATH.exists()
    ep_df, act_df, meta = DataLoader.load_from_file(SAMPLE_CSV_PATH)
    assert not ep_df.empty
    assert len(ep_df) >= 100
    assert "episode_id" in ep_df.columns
    assert "score_a" in ep_df.columns
    assert "winner" in ep_df.columns


def test_generate_sample_dataset():
    ep_df, act_df = DataLoader.generate_sample_dataset()
    assert len(ep_df) == 120
    assert len(act_df) > 0
    assert "is_sample_data" in ep_df.columns
