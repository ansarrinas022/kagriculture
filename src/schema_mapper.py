"""
Flexible Schema Mapping Layer for Kaggriculture Datasets.
Maps varied column names, nested structures, and Kaggle simulation replay JSONs
into canonical analytical schemas without losing original data.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


# Common alias mappings for standard episode attributes
ALIAS_MAP: Dict[str, List[str]] = {
    "episode_id": ["episode_id", "episodeid", "id", "match_id", "game_id", "simulation_id"],
    "agent_a": ["agent_a", "agent_1", "player_1", "team_1", "bot_a", "agent1", "player1", "submission_1"],
    "agent_b": ["agent_b", "agent_2", "player_2", "team_2", "bot_b", "agent2", "player2", "submission_2"],
    "score_a": ["score_a", "score_1", "bank_a", "bank_1", "final_score_a", "reward_1", "score1"],
    "score_b": ["score_b", "score_2", "bank_b", "bank_2", "final_score_b", "reward_2", "score2"],
    "winner": ["winner", "winning_agent", "outcome", "result", "victor"],
    "total_turns": ["total_turns", "turns", "steps", "total_steps", "step_count", "rounds"],
    "season_day": ["season_day", "day", "days_simulated", "game_day"],
    "actions_a_count": ["actions_a_count", "actions_1_count", "actions_a", "total_actions_a"],
    "actions_b_count": ["actions_b_count", "actions_2_count", "actions_b", "total_actions_b"],
    "crops_harvested_a": ["crops_harvested_a", "harvests_a", "crops_a", "harvested_a"],
    "crops_harvested_b": ["crops_harvested_b", "harvests_b", "crops_b", "harvested_b"],
    "market_sales_a": ["market_sales_a", "sales_a", "market_revenue_a", "revenue_a"],
    "market_sales_b": ["market_sales_b", "sales_b", "market_revenue_b", "revenue_b"],
    "land_quadrants_a": ["land_quadrants_a", "quadrants_a", "land_a", "tiles_a"],
    "land_quadrants_b": ["land_quadrants_b", "quadrants_b", "land_b", "tiles_b"],
    "livestock_count_a": ["livestock_count_a", "animals_a", "livestock_a"],
    "livestock_count_b": ["livestock_count_b", "animals_b", "livestock_b"],
    "status": ["status", "state", "game_state", "completion_status"]
}


class SchemaMapper:
    """Detects, maps, and validates dataset schemas against canonical fields."""

    @staticmethod
    def detect_column_mappings(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Inspects DataFrame columns and finds the best match for each canonical field.
        Case-insensitive matching with normalization.
        """
        existing_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}
        mapping: Dict[str, Optional[str]] = {}

        for canonical_name, aliases in ALIAS_MAP.items():
            matched_col = None
            for alias in aliases:
                norm_alias = alias.lower().strip().replace(" ", "_")
                if norm_alias in existing_cols:
                    matched_col = existing_cols[norm_alias]
                    break
            mapping[canonical_name] = matched_col

        return mapping

    @staticmethod
    def apply_mapping(df: pd.DataFrame, custom_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Applies mapping from source columns to canonical names.
        Preserves all original columns and creates canonical aliases.
        """
        mapped_df = df.copy()
        mappings = custom_mapping or SchemaMapper.detect_column_mappings(df)

        for canonical_name, source_col in mappings.items():
            if source_col and source_col in df.columns:
                if canonical_name != source_col:
                    mapped_df[canonical_name] = df[source_col]

        # Auto-compute winner if missing but scores are present
        if "winner" not in mapped_df.columns or mapped_df["winner"].isna().all():
            if "score_a" in mapped_df.columns and "score_b" in mapped_df.columns:
                def derive_winner(row):
                    sa = row.get("score_a")
                    sb = row.get("score_b")
                    if pd.isna(sa) or pd.isna(sb):
                        return None
                    try:
                        sa_f = float(sa)
                        sb_f = float(sb)
                        if sa_f > sb_f:
                            return row.get("agent_a", "Agent A")
                        elif sb_f > sa_f:
                            return row.get("agent_b", "Agent B")
                        else:
                            return "Tie"
                    except (ValueError, TypeError):
                        return None
                mapped_df["winner"] = mapped_df.apply(derive_winner, axis=1)

        # Standardize episode_id if available
        if "episode_id" in mapped_df.columns:
            mapped_df["episode_id"] = mapped_df["episode_id"].astype(str)

        return mapped_df

    @staticmethod
    def parse_kaggle_replay_json(json_data: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses Kaggle simulation replay JSON structure.
        Kaggle JSON replays typically feature:
        - metadata / description / specification / configuration
        - steps: list of steps, each containing agents' actions, rewards, observations, and status.
        
        Returns:
            episodes_df: Single-row or multi-row DataFrame of episode-level summary.
            actions_df: Granular step-by-step actions DataFrame.
        """
        episodes_list: List[Dict[str, Any]] = []
        actions_list: List[Dict[str, Any]] = []

        # If data is a list of episodes
        raw_episodes = json_data if isinstance(json_data, list) else [json_data]

        for ep_idx, ep in enumerate(raw_episodes):
            if not isinstance(ep, dict):
                continue

            ep_id = str(ep.get("id") or ep.get("episode_id") or f"ep_{ep_idx + 1}")
            agents = ep.get("agents") or ep.get("players") or []
            
            # Extract agent names if present
            agent_a = "Agent_1"
            agent_b = "Agent_2"
            if isinstance(agents, list) and len(agents) >= 2:
                agent_a = str(agents[0].get("name", agents[0]) if isinstance(agents[0], dict) else agents[0])
                agent_b = str(agents[1].get("name", agents[1]) if isinstance(agents[1], dict) else agents[1])
            elif "agent_a" in ep and "agent_b" in ep:
                agent_a = str(ep["agent_a"])
                agent_b = str(ep["agent_b"])

            steps = ep.get("steps") or []
            final_score_a = None
            final_score_b = None
            total_turns = len(steps)

            # Analyze steps
            actions_a_count = 0
            actions_b_count = 0
            harvest_a = 0
            harvest_b = 0
            sales_a = 0.0
            sales_b = 0.0

            for step_idx, step in enumerate(steps):
                step_num = step_idx
                # Step format in Kaggle: list of agent states [agent_0_state, agent_1_state]
                if isinstance(step, list) and len(step) >= 2:
                    st_a = step[0] if isinstance(step[0], dict) else {}
                    st_b = step[1] if isinstance(step[1], dict) else {}

                    act_a = st_a.get("action")
                    act_b = st_b.get("action")

                    rew_a = st_a.get("reward")
                    rew_b = st_b.get("reward")

                    if rew_a is not None:
                        final_score_a = rew_a
                    if rew_b is not None:
                        final_score_b = rew_b

                    if act_a:
                        actions_a_count += 1
                        act_str = str(act_a.get("type", act_a) if isinstance(act_a, dict) else act_a)
                        if "HARVEST" in act_str.upper():
                            harvest_a += 1
                        if "SELL" in act_str.upper():
                            sales_a += float(act_a.get("amount", 10)) if isinstance(act_a, dict) else 10.0
                        actions_list.append({
                            "episode_id": ep_id,
                            "turn": step_num,
                            "agent": agent_a,
                            "agent_role": "Agent A",
                            "action": act_str,
                            "reward_after": rew_a,
                            "action_details": act_a if isinstance(act_a, dict) else {"raw": str(act_a)}
                        })

                    if act_b:
                        actions_b_count += 1
                        act_str = str(act_b.get("type", act_b) if isinstance(act_b, dict) else act_b)
                        if "HARVEST" in act_str.upper():
                            harvest_b += 1
                        if "SELL" in act_str.upper():
                            sales_b += float(act_b.get("amount", 10)) if isinstance(act_b, dict) else 10.0
                        actions_list.append({
                            "episode_id": ep_id,
                            "turn": step_num,
                            "agent": agent_b,
                            "agent_role": "Agent B",
                            "action": act_str,
                            "reward_after": rew_b,
                            "action_details": act_b if isinstance(act_b, dict) else {"raw": str(act_b)}
                        })
                elif isinstance(step, dict):
                    # Flat step record
                    actions_list.append({
                        "episode_id": ep_id,
                        "turn": step.get("turn", step_idx),
                        "agent": step.get("agent", agent_a),
                        "action": step.get("action", "REST_OR_IDLE"),
                        "reward_after": step.get("reward"),
                        "action_details": step
                    })

            # Check top-level scores if not from steps
            if final_score_a is None:
                final_score_a = ep.get("score_a", ep.get("final_score_a", ep.get("reward_a")))
            if final_score_b is None:
                final_score_b = ep.get("score_b", ep.get("final_score_b", ep.get("reward_b")))

            # Determine winner
            winner = None
            if final_score_a is not None and final_score_b is not None:
                try:
                    fsa = float(final_score_a)
                    fsb = float(final_score_b)
                    if fsa > fsb:
                        winner = agent_a
                    elif fsb > fsa:
                        winner = agent_b
                    else:
                        winner = "Tie"
                except (ValueError, TypeError):
                    winner = ep.get("winner")

            episodes_list.append({
                "episode_id": ep_id,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "score_a": final_score_a,
                "score_b": final_score_b,
                "winner": winner,
                "total_turns": total_turns if total_turns > 0 else ep.get("total_turns", 720),
                "season_day": ep.get("season_day", 30),
                "actions_a_count": actions_a_count or ep.get("actions_a_count", 0),
                "actions_b_count": actions_b_count or ep.get("actions_b_count", 0),
                "crops_harvested_a": harvest_a or ep.get("crops_harvested_a", 0),
                "crops_harvested_b": harvest_b or ep.get("crops_harvested_b", 0),
                "market_sales_a": sales_a or ep.get("market_sales_a", 0.0),
                "market_sales_b": sales_b or ep.get("market_sales_b", 0.0),
                "land_quadrants_a": ep.get("land_quadrants_a", 2),
                "land_quadrants_b": ep.get("land_quadrants_b", 2),
                "livestock_count_a": ep.get("livestock_count_a", 4),
                "livestock_count_b": ep.get("livestock_count_b", 4),
                "status": ep.get("status", "FINISHED")
            })

        episodes_df = pd.DataFrame(episodes_list)
        actions_df = pd.DataFrame(actions_list)
        return episodes_df, actions_df
