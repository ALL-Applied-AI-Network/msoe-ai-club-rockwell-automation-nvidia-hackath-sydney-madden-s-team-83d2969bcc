"""Inverse Recipe Regressor: maps observed fluid optical properties to physical tank volumes.

Combines optical colorimetry (Beer-Lambert subtractive dye mixing) with machine learning.
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from src.config import LABELS_DIR, IMAGES_DIR, OUTPUT_DIR, RECIPES
from src.vision.volume_estimator import VolumeEstimator
from src.vision.color_extractor import ColorExtractor


class CompositionModel:
    """Predicts physical fluid volumes (T100, T200, T400 in mL) from optical features."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (OUTPUT_DIR / "composition_model.pkl")
        self.volume_estimator = VolumeEstimator()
        self.color_extractor = ColorExtractor()

    def estimate_composition(
        self,
        vol_info: Dict[str, Any],
        color_info: Dict[str, Any],
    ) -> Dict[str, float]:
        """Physics-informed optical colorimeter.

        Uses subtractive color mixing laws to decompose the observed liquid into:
        - T100 (Clear diluent)
        - T200 (Orange dye)
        - T400 (Blue dye)
        """
        vol = float(vol_info["volume_ml"])
        hue = float(color_info["hue"])
        sat = float(color_info["saturation"])
        r_norm = float(color_info["r_norm"])
        b_norm = float(color_info["b_norm"])

        # Reference anchor points for pure dyes in the station
        HUE_ORANGE = 22.0
        HUE_BLUE = 103.0

        # Case 1: Pure Blue (Hue >= 90, dominant blue channel)
        if hue >= 90.0 or (b_norm > 0.45 and r_norm < 0.15):
            t200 = 0.0
            # Higher saturation means higher dye-to-diluent concentration
            dye_frac = min(0.85, max(0.20, sat / 255.0 * 0.90))
            t400 = round(vol * dye_frac, 1)
            t100 = round(max(0.0, vol - t400), 1)
            return {"t100_ml": t100, "t200_ml": 0.0, "t400_ml": t400}

        # Case 2: Pure Orange (Hue <= 32, dominant red/orange channel)
        if hue <= 32.0 or (r_norm > 0.45 and b_norm < 0.15):
            t400 = 0.0
            dye_frac = min(0.85, max(0.20, sat / 255.0 * 0.85))
            t200 = round(vol * dye_frac, 1)
            t100 = round(max(0.0, vol - t200), 1)
            return {"t100_ml": t100, "t200_ml": t200, "t400_ml": 0.0}

        # Case 3: Green / Mixed (Residual orange + blue, or intentional triple recipe)
        hue_clamped = float(np.clip(hue, HUE_ORANGE, HUE_BLUE))
        blue_weight = (hue_clamped - HUE_ORANGE) / (HUE_BLUE - HUE_ORANGE)
        orange_weight = 1.0 - blue_weight

        dye_frac = min(0.85, max(0.20, sat / 220.0 * 0.80))
        total_dye_ml = vol * dye_frac

        t200 = round(total_dye_ml * orange_weight, 1)
        t400 = round(total_dye_ml * blue_weight, 1)
        t100 = round(max(0.0, vol - t200 - t400), 1)

        return {"t100_ml": t100, "t200_ml": t200, "t400_ml": t400}

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Infers composition from an unseen frame."""
        vol_info = self.volume_estimator.estimate_volume(frame)
        color_info = self.color_extractor.extract_liquid_color(frame, vol_info["y_meniscus"])

        comp = self.estimate_composition(vol_info, color_info)

        return {
            "pred_t100_ml": comp["t100_ml"],
            "pred_t200_ml": comp["t200_ml"],
            "pred_t400_ml": comp["t400_ml"],
            "pred_total_ml": vol_info["volume_ml"],
            "color_info": color_info,
            "vol_info": vol_info,
        }
