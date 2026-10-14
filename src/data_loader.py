"""
Data Loader Module for Kaggriculture.
Supports CSV, JSON, JSONL, Parquet, and multi-file ingestion.
Provides file inspection, memory-efficient loading, Kaggle API integration,
and sample dataset generation.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import os
import io
import pandas as pd
import numpy as np

from config import DATA_DIR, SAMPLE_CSV_PATH, SAMPLE_JSON_PATH
from src.schema_mapper import SchemaMapper


class DataLoader:
    """Handles dataset loading, multi-format parsing, and Kaggle download integration."""

    @staticmethod
    def load_from_file(file_obj_or_path: Union[str, Path, io.BytesIO], file_type: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Loads dataset from file path or file-like buffer (Streamlit UploadedFile).
        Returns:
            episodes_df: pd.DataFrame
            actions_df: Optional[pd.DataFrame] (if replay data is available)
            metadata: Dict[str, Any] (file info, format, raw dimensions)
        """
        filename = getattr(file_obj_or_path, "name", str(file_obj_or_path))
        ext = (file_type or Path(filename).suffix.lstrip(".")).lower()
        metadata: Dict[str, Any] = {
            "source_filename": filename,
            "format": ext,
            "loaded_via": "file_upload"
        }

        episodes_df = pd.DataFrame()
        actions_df = None

        if ext == "csv":
            df_raw = pd.read_csv(file_obj_or_path)
            metadata["raw_shape"] = df_raw.shape
            episodes_df = SchemaMapper.apply_mapping(df_raw)

        elif ext in ["json", "jsonl"]:
            # Check if JSON is a Kaggle simulation replay or flat JSON
            if hasattr(file_obj_or_path, "read"):
                raw_bytes = file_obj_or_path.read()
                file_obj_or_path.seek(0)
                try:
                    raw_str = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raw_str = raw_bytes.decode("latin-1")
            else:
                with open(file_obj_or_path, "r", encoding="utf-8") as f:
                    raw_str = f.read()

            try:
                # Try parsing as standard JSON
                parsed_json = json.loads(raw_str)
                if isinstance(parsed_json, dict) and ("steps" in parsed_json or "configuration" in parsed_json):
                    # Single Kaggle simulation replay
                    episodes_df, actions_df = SchemaMapper.parse_kaggle_replay_json(parsed_json)
                elif isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict) and "steps" in parsed_json[0]:
                    # Multi-episode Kaggle simulation replays
                    episodes_df, actions_df = SchemaMapper.parse_kaggle_replay_json(parsed_json)
                else:
                    # Tabular JSON list of records
                    df_raw = pd.DataFrame(parsed_json)
                    episodes_df = SchemaMapper.apply_mapping(df_raw)
            except json.JSONDecodeError:
                # Try parsing as JSONL (line-delimited JSON)
                lines = raw_str.strip().splitlines()
                json_records = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            json_records.append(json.loads(line))
                        except Exception:
                            continue
                if json_records:
                    if "steps" in json_records[0]:
                        episodes_df, actions_df = SchemaMapper.parse_kaggle_replay_json(json_records)
                    else:
                        df_raw = pd.DataFrame(json_records)
                        episodes_df = SchemaMapper.apply_mapping(df_raw)
                else:
                    raise ValueError(f"Unable to parse {filename} as JSON or JSON Lines.")

            metadata["raw_shape"] = episodes_df.shape

        elif ext == "parquet":
            df_raw = pd.read_parquet(file_obj_or_path)
            metadata["raw_shape"] = df_raw.shape
            episodes_df = SchemaMapper.apply_mapping(df_raw)

        else:
            raise ValueError(f"Unsupported file format '.{ext}'. Supported formats are CSV, JSON, JSONL, Parquet.")

        metadata["episodes_count"] = len(episodes_df)
        metadata["has_actions"] = actions_df is not None and not actions_df.empty
        if metadata["has_actions"]:
            metadata["actions_count"] = len(actions_df)

        return episodes_df, actions_df, metadata

    @staticmethod
    def load_multiple_files(file_list: List[Any]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Combines multiple compatible uploaded files into a unified dataset.
        """
        all_episodes = []
        all_actions = []
        filenames = []

        for f in file_list:
            ep_df, act_df, meta = DataLoader.load_from_file(f)
            if not ep_df.empty:
                all_episodes.append(ep_df)
            if act_df is not None and not act_df.empty:
                all_actions.append(act_df)
            filenames.append(meta.get("source_filename", "unknown"))

        if not all_episodes:
            return pd.DataFrame(), None, {"loaded_via": "multiple_files", "count": 0}

        combined_episodes = pd.concat(all_episodes, ignore_index=True)
        # Deduplicate by episode_id if available
        if "episode_id" in combined_episodes.columns:
            combined_episodes = combined_episodes.drop_duplicates(subset=["episode_id"], keep="first")

        combined_actions = pd.concat(all_actions, ignore_index=True) if all_actions else None

        metadata = {
            "source_filenames": filenames,
            "format": "multiple_files",
            "episodes_count": len(combined_episodes),
            "has_actions": combined_actions is not None and not combined_actions.empty,
            "actions_count": len(combined_actions) if combined_actions is not None else 0
        }

        return combined_episodes, combined_actions, metadata

    @staticmethod
    def download_from_kaggle(dataset_ref: str = "kaggle/kaggriculture-episodes-2026-09-08", download_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Attempts to download official dataset using Kaggle API if credentials exist.
        Never hardcodes credentials. Uses system environment or ~/.kaggle/kaggle.json.
        """
        target_dir = download_dir or DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=True)
            return True, f"Successfully downloaded and extracted {dataset_ref} to {target_dir}"
        except ImportError:
            return False, "The 'kaggle' package is not installed. Run 'pip install kaggle' or configure ~/.kaggle/kaggle.json."
        except Exception as e:
            return False, f"Kaggle API download failed: {str(e)}. Please check your credentials or upload files manually."

    @staticmethod
    def generate_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates realistic, statistically diverse sample Kaggriculture episodes and replays
        specifically designed for demonstrations and automated testing when the Kaggle dataset is not uploaded.
        Clearly designated as SAMPLE DATA.
        """
        np.random.seed(42)
        n_episodes = 120

        agents_pool = [
            "AgriBot-Alpha",
            "HarvestKing-v2",
            "FarmMaster-RL",
            "CropOptima-DeepQ",
            "AgroEcon-PPO",
            "GreenThumb-Agent",
            "TerraTactician",
            "BioEconomic-AI"
        ]

        crops = ["WHEAT", "CORN", "CARROT", "POTATO"]
        episodes_data = []
        actions_data = []

        for i in range(1, n_episodes + 1):
            ep_id = f"ep_{i:04d}"
            # Choose 2 distinct agents
            ag_a, ag_b = np.random.choice(agents_pool, size=2, replace=False)

            # Assign skill/strategy biases to agents
            bias_a = 1.15 if "RL" in ag_a or "Alpha" in ag_a else (0.95 if "Green" in ag_a else 1.0)
            bias_b = 1.15 if "RL" in ag_b or "Alpha" in ag_b else (0.95 if "Green" in ag_b else 1.0)

            # Simulation attributes
            turns = 720  # standard 30 days * 24 turns
            season_day = 30

            # Base actions counts
            act_a_count = int(np.random.normal(520, 45) * bias_a)
            act_b_count = int(np.random.normal(510, 50) * bias_b)

            # Harvests & Sales
            harvest_a = int(np.random.normal(160, 25) * bias_a)
            harvest_b = int(np.random.normal(155, 30) * bias_b)
            sales_a = round(float(np.random.normal(12500, 1800) * bias_a), 2)
            sales_b = round(float(np.random.normal(12200, 2000) * bias_b), 2)

            # Final scores (Bank balance at end of season)
            score_a = round(sales_a - np.random.normal(2500, 300) + (harvest_a * 15), 2)
            score_b = round(sales_b - np.random.normal(2500, 300) + (harvest_b * 15), 2)

            # Determine winner
            if score_a > score_b:
                winner = ag_a
            elif score_b > score_a:
                winner = ag_b
            else:
                winner = "Tie"

            quads_a = int(np.random.choice([1, 2, 3, 4], p=[0.1, 0.5, 0.3, 0.1]))
            quads_b = int(np.random.choice([1, 2, 3, 4], p=[0.1, 0.5, 0.3, 0.1]))
            livestock_a = int(np.random.choice([0, 2, 4, 6, 8], p=[0.15, 0.3, 0.3, 0.15, 0.1]))
            livestock_b = int(np.random.choice([0, 2, 4, 6, 8], p=[0.15, 0.3, 0.3, 0.15, 0.1]))

            episodes_data.append({
                "episode_id": ep_id,
                "agent_a": ag_a,
                "agent_b": ag_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "total_turns": turns,
                "season_day": season_day,
                "actions_a_count": act_a_count,
                "actions_b_count": act_b_count,
                "crops_harvested_a": harvest_a,
                "crops_harvested_b": harvest_b,
                "market_sales_a": sales_a,
                "market_sales_b": sales_b,
                "land_quadrants_a": quads_a,
                "land_quadrants_b": quads_b,
                "livestock_count_a": livestock_a,
                "livestock_count_b": livestock_b,
                "status": "FINISHED",
                "is_sample_data": True
            })

            # Generate granular replay step actions for the first 15 episodes for deep exploration
            if i <= 15:
                curr_bank_a = 500.0
                curr_bank_b = 500.0
                for turn in range(1, 41):  # Sample first 40 turns of detailed replay
                    # Agent A action
                    act_type_a = np.random.choice([
                        "PLANT_WHEAT", "PLANT_CORN", "WATER_CROPS", "FERTILIZE_CROPS", 
                        "HARVEST_CROPS", "SELL_MARKET", "BUY_SEEDS", "FEED_LIVESTOCK"
                    ], p=[0.25, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05])
                    
                    if "SELL" in act_type_a:
                        curr_bank_a += 250.0
                    elif "BUY" in act_type_a:
                        curr_bank_a -= 80.0
                    elif "HARVEST" in act_type_a:
                        curr_bank_a += 50.0

                    actions_data.append({
                        "episode_id": ep_id,
                        "turn": turn,
                        "agent": ag_a,
                        "agent_role": "Agent A",
                        "action": act_type_a,
                        "reward_after": round(curr_bank_a, 2),
                        "action_details": {"crop": np.random.choice(crops) if "PLANT" in act_type_a else None}
                    })

                    # Agent B action
                    act_type_b = np.random.choice([
                        "PLANT_WHEAT", "PLANT_CARROT", "WATER_CROPS", "FEED_LIVESTOCK",
                        "HARVEST_CROPS", "SELL_MARKET", "BUY_QUADRANT", "REST_OR_IDLE"
                    ], p=[0.20, 0.25, 0.20, 0.10, 0.10, 0.05, 0.02, 0.08])

                    if "SELL" in act_type_b:
                        curr_bank_b += 240.0
                    elif "BUY" in act_type_b:
                        curr_bank_b -= 150.0
                    elif "HARVEST" in act_type_b:
                        curr_bank_b += 45.0

                    actions_data.append({
                        "episode_id": ep_id,
                        "turn": turn,
                        "agent": ag_b,
                        "agent_role": "Agent B",
                        "action": act_type_b,
                        "reward_after": round(curr_bank_b, 2),
                        "action_details": {"crop": "CARROT" if "CARROT" in act_type_b else "WHEAT"}
                    })

        ep_df = pd.DataFrame(episodes_data)
        act_df = pd.DataFrame(actions_data)
        return ep_df, act_df
