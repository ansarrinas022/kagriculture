"""
Kaggriculture - Configuration & Design Constants
Agricultural Simulation Analysis & Strategy Platform
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV_PATH = DATA_DIR / "kaggriculture_sample_episodes.csv"
SAMPLE_JSON_PATH = DATA_DIR / "kaggriculture_sample_replays.json"

# Application Metadata
APP_TITLE = "Kaggriculture"
APP_SUBTITLE = "Agricultural Simulation Analysis & Strategy Platform"
APP_ICON = "🌱"
VERSION = "1.0.0"

# Design Tokens (Agricultural Dark Green & Emerald Theme)
THEME_COLORS = {
    "bg_dark": "#0B1914",
    "bg_card": "#132A22",
    "bg_card_secondary": "#1A382E",
    "bg_light_cream": "#F4F6F0",
    "accent_emerald": "#10B981",
    "accent_mint": "#34D399",
    "accent_gold": "#F59E0B",
    "accent_wheat": "#EAB308",
    "accent_terra": "#D97706",
    "accent_danger": "#EF4444",
    "accent_water": "#38BDF8",
    "text_primary": "#ECFDF5",
    "text_secondary": "#9CA3AF",
    "text_muted": "#64748B",
    "border_color": "#1E3E33",
}

# Standard Kaggriculture Action Types
KNOWN_ACTION_TYPES = [
    "PLANT_WHEAT",
    "PLANT_CORN",
    "PLANT_CARROT",
    "PLANT_POTATO",
    "WATER_CROPS",
    "FERTILIZE_CROPS",
    "HARVEST_CROPS",
    "FEED_LIVESTOCK",
    "MILK_COWS",
    "COLLECT_EGGS",
    "BUY_SEEDS",
    "BUY_FEED",
    "BUY_QUADRANT",
    "HIRE_FARMHAND",
    "SELL_MARKET",
    "WEED_FIELD",
    "REST_OR_IDLE"
]

# Action Groupings for Analytics
ACTION_CATEGORIES = {
    "Crop Operations": ["PLANT_WHEAT", "PLANT_CORN", "PLANT_CARROT", "PLANT_POTATO", "WATER_CROPS", "FERTILIZE_CROPS", "HARVEST_CROPS", "WEED_FIELD"],
    "Livestock Operations": ["FEED_LIVESTOCK", "MILK_COWS", "COLLECT_EGGS"],
    "Commerce & Trading": ["SELL_MARKET", "BUY_SEEDS", "BUY_FEED"],
    "Farm Expansion": ["BUY_QUADRANT", "HIRE_FARMHAND"],
    "Idle / Pass": ["REST_OR_IDLE"]
}

# Canonical Episode Schema Expected Fields
CANONICAL_COLUMNS = [
    "episode_id",
    "agent_a",
    "agent_b",
    "score_a",
    "score_b",
    "winner",
    "total_turns",
    "season_day",
    "actions_a_count",
    "actions_b_count",
    "crops_harvested_a",
    "crops_harvested_b",
    "market_sales_a",
    "market_sales_b",
    "land_quadrants_a",
    "land_quadrants_b",
    "livestock_count_a",
    "livestock_count_b",
    "status"
]

# Custom CSS for Streamlit UI
CUSTOM_CSS = """
<style>
    /* Main Background & Base Styling */
    .stApp {
        background-color: #0B1914;
        color: #ECFDF5;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(11, 25, 20, 0.85);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #08130F;
        border-right: 1px solid #1A382E;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #34D399;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #132A22;
        border: 1px solid #1E3E33;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #10B981;
    }
    div[data-testid="metric-container"] label {
        color: #9CA3AF !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #10B981 !important;
        font-weight: 700 !important;
        font-size: 1.85rem !important;
    }
    
    /* Custom Card Containers */
    .kag-card {
        background: #132A22;
        border: 1px solid #1E3E33;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .kag-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        border-bottom: 1px solid #1E3E33;
        padding-bottom: 10px;
    }
    
    .kag-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #1A382E;
        color: #34D399;
        border: 1px solid #10B981;
    }
    
    .kag-badge-gold {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }
    
    .kag-badge-info {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid #0284C7;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 100%);
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.45);
        transform: translateY(-1px);
        color: #FFFFFF;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1E3E33;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #9CA3AF;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #34D399 !important;
        background-color: #132A22 !important;
        border-bottom: 2px solid #10B981 !important;
        font-weight: 600;
    }
    
    /* Dataframe table styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E3E33;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox select, .stMultiSelect {
        background-color: #0E221B !important;
        border: 1px solid #1E3E33 !important;
        color: #ECFDF5 !important;
        border-radius: 8px !important;
    }
</style>
"""
