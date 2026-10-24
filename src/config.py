"""Configuration constants and paths for Fluid Mixing Station 3."""

from pathlib import Path
from typing import Dict, Any

# --- Base Directory Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAL_DATA_DIR = REPO_ROOT / "personal_data"
IMAGES_DIR = PERSONAL_DATA_DIR / "images"
VIDEOS_DIR = PERSONAL_DATA_DIR / "videos"
LABELS_DIR = PERSONAL_DATA_DIR / "labels"
OUTPUT_DIR = REPO_ROOT / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- Standard Camera Frame Dimensions ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Regions of Interest (ROI) [x, y, width, height] ---
# Calibrated for the stationary vial_fill camera view
NOZZLE_ROI = (295, 70, 50, 60)       # Nozzle tip region (detects stream/flow)
VIAL_BODY_ROI = (275, 150, 90, 260)  # Main vial cylinder (for meniscus & fluid color)
CART_TRACK_ROI = (250, 400, 140, 70) # Conveyor cart wheels/bracket (motion detection)

# --- Liquid Meniscus Height Calibration (Video 5) ---
# Volume (mL) = SLOPE * (Y_VIAL_BOTTOM - y_meniscus) + INTERCEPT
# The cylindrical vial has constant diameter; volume scales linearly with fluid height.
MENISCUS_CALIBRATION = {
    "y_bottom": 410,       # Pixel Y coordinate of the inner bottom of the vial
    "px_per_ml": 2.65,     # Calibrated pixels of height per mL of liquid
    "min_valid_ml": 0.0,
    "max_valid_ml": 85.0,  # Physical vial max limit before overflow is ~80-85 mL
}

# --- Tank Definitions ---
TANKS = {
    "T100": {"name": "T-100", "color": "clear", "role": "diluent", "position": 1},
    "T200": {"name": "T-200", "color": "orange", "role": "dye", "position": 2},
    "T300": {"name": "T-300", "color": "disabled", "role": "out_of_service", "position": 3},
    "T400": {"name": "T-400", "color": "blue", "role": "dye", "position": 4},
}

# --- Known Recipes (Ground Truth from Video 5) ---
RECIPES: Dict[int, Dict[str, Any]] = {
    5: {
        "recipe_id": 5, "name": "Triple Low Even", "type": "Triple",
        "t100_ml": 10, "t200_ml": 10, "t400_ml": 10, "total_ml": 30,
        "dispense_order": "T200>T400>T100", "target_color": "green",
    },
    6: {
        "recipe_id": 6, "name": "Pair 100+200 Pale Orange", "type": "Pair 100+200",
        "t100_ml": 30, "t200_ml": 10, "t400_ml": 0, "total_ml": 40,
        "dispense_order": "T200>T100", "target_color": "pale orange",
    },
    7: {
        "recipe_id": 7, "name": "Pair 100+400 Pale Blue", "type": "Pair 100+400",
        "t100_ml": 20, "t200_ml": 0, "t400_ml": 50, "total_ml": 70,
        "dispense_order": "T400>T100", "target_color": "blue",
    },
    8: {
        "recipe_id": 8, "name": "Triple Even", "type": "Triple",
        "t100_ml": 20, "t200_ml": 20, "t400_ml": 20, "total_ml": 60,
        "dispense_order": "T200>T400>T100", "target_color": "green",
    },
    9: {
        "recipe_id": 9, "name": "Triple Dark Saturated", "type": "Triple",
        "t100_ml": 10, "t200_ml": 35, "t400_ml": 35, "total_ml": 80,
        "dispense_order": "T200>T400>T100", "target_color": "dark green",
    },
    10: {
        "recipe_id": 10, "name": "Triple Pale Wash", "type": "Triple",
        "t100_ml": 45, "t200_ml": 15, "t400_ml": 15, "total_ml": 75,
        "dispense_order": "T200>T400>T100", "target_color": "pale green",
    },
    11: {
        "recipe_id": 11, "name": "Triple High Blue", "type": "Triple",
        "t100_ml": 10, "t200_ml": 10, "t400_ml": 50, "total_ml": 70,
        "dispense_order": "T200>T400>T100", "target_color": "cyan/green",
    },
    12: {
        "recipe_id": 12, "name": "Triple High Orange", "type": "Triple",
        "t100_ml": 10, "t200_ml": 50, "t400_ml": 10, "total_ml": 70,
        "dispense_order": "T200>T400>T100", "target_color": "olive/brown",
    },
    13: {
        "recipe_id": 13, "name": "Pair 200+400 No Diluent", "type": "Pair 200+400",
        "t100_ml": 0, "t200_ml": 50, "t400_ml": 20, "total_ml": 70,
        "dispense_order": "T200>T400", "target_color": "dark green",
    },
    14: {
        "recipe_id": 14, "name": "Pair 100+400 High Diluent", "type": "Pair 100+400",
        "t100_ml": 60, "t200_ml": 0, "t400_ml": 10, "total_ml": 70,
        "dispense_order": "T400>T100", "target_color": "pale blue",
    },
}
