"""
Unit tests for StatisticalAnalysis and AgentComparison.
"""
import pytest
import pandas as pd
import numpy as np
from src.statistical_analysis import StatisticalAnalysis
from src.agent_comparison import AgentComparison


def test_statistical_analysis():
    df = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 1000],
        "harvests": [10, 20, 30, 40, 50, 100]
    })
    stats = StatisticalAnalysis(df)
    desc = stats.compute_descriptive_stats(["score", "harvests"])
    assert len(desc) == 2
    assert desc[desc["Feature"] == "score"]["Mean"].iloc[0] > 0

    corr, note = stats.compute_correlation_matrix(["score", "harvests"])
    assert not corr.empty
    assert "causation" in note.lower()

    outliers = stats.detect_outliers("score", method="iqr")
    assert "total_outliers" in outliers


def test_agent_comparison_matched_and_global():
    df = pd.DataFrame({
        "episode_id": ["e1", "e2", "e3"],
        "agent_a": ["Agent_X", "Agent_Y", "Agent_X"],
        "agent_b": ["Agent_Y", "Agent_X", "Agent_Z"],
        "score_a": [1500, 800, 1200],
        "score_b": [1200, 1400, 1100],
        "winner": ["Agent_X", "Agent_Y", "Agent_X"]
    })
    comp = AgentComparison(df)
    agents = comp.get_all_agents()
    assert "Agent_X" in agents
    assert "Agent_Y" in agents

    matched = comp.get_matched_episodes("Agent_X", "Agent_Y")
    assert len(matched) == 2  # e1 and e2

    res = comp.compare_agents("Agent_X", "Agent_Y")
    assert res["matched_stats"]["matched_episodes_count"] == 2
    assert res["matched_stats"]["wins_a"] == 1  # Agent_X won in e1
    assert res["matched_stats"]["wins_b"] == 1  # Agent_Y won in e2

