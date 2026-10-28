"""Liquid level (meniscus) detection and volume estimation in milliliters."""

from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import cv2
import numpy as np

from src.config import (
    VIAL_BODY_ROI,
    MENISCUS_CALIBRATION,
)


class VolumeEstimator:
    """Estimates fluid volume (mL) from the liquid meniscus height in the vial."""

    # Calibrated camera perspective projection model parameters:
    # V(px) = a * px^2 + b * px + c
    # Derived from pinhole camera projective trigonometry on a tilted vertical cylinder
    # (captures curved base offset, linear column barrel, and camera tilt foreshortening).
    PERSPECTIVE_REGRESSION_COEFFS = [
        1.00000000e-05,
        3.42711864e-01,
        1.00000000e+00,
    ]

    def __init__(
        self,
        y_bottom: int = MENISCUS_CALIBRATION["y_bottom"],
        px_per_ml: float = MENISCUS_CALIBRATION["px_per_ml"],
    ):
        self.y_bottom = y_bottom
        self.px_per_ml = px_per_ml
        self.vial_roi = VIAL_BODY_ROI

    def detect_meniscus_y(self, frame: np.ndarray) -> Tuple[int, float]:
        """Detect the liquid meniscus row (in full frame Y coordinates).

        Returns:
            Tuple[int, float]: (y_meniscus_full_frame, confidence_score)
        """
        vx, vy, vw, vh = self.vial_roi
        margin_x = int(vw * 0.25)
        crop_x = vx + margin_x
        crop_w = vw - (2 * margin_x)
        crop = frame[vy : vy + vh, crop_x : crop_x + crop_w]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel = np.abs(sobel_y)
        row_gradients = np.mean(abs_sobel, axis=1)

        search_start_idx = int(vh * 0.28)
        search_end_idx = vh - 10

        valid_gradients = row_gradients[search_start_idx:search_end_idx]
        if len(valid_gradients) == 0:
            return self.y_bottom, 0.0

        best_sub_idx = int(np.argmax(valid_gradients))
        best_crop_y = search_start_idx + best_sub_idx
        y_full_frame = vy + best_crop_y
        max_grad = float(valid_gradients[best_sub_idx])

        mean_grad = float(np.mean(valid_gradients)) + 1e-6
        confidence = min(1.0, max_grad / (3.0 * mean_grad))

        if y_full_frame > self.y_bottom - 5:
            y_full_frame = self.y_bottom

        return y_full_frame, confidence

    def count_fluid_pixels(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> int:
        """Counts fluid pixels along the vertical center tube column relative to empty ref frame.

        Accounts for the physical clamp obstruction at y in [242, 265] with strict physical
        continuity verification immediately above the clamp (y in [230, 238]).
        """
        vial_x1, vial_x2 = 306, 316
        y_top, y_bot = 160, self.y_bottom + 2

        cur_crop = frame[y_top:y_bot, vial_x1:vial_x2]
        ref_crop = ref_frame[y_top:y_bot, vial_x1:vial_x2]

        diff = np.abs(cur_crop.astype(float) - ref_crop.astype(float)).mean(axis=2).mean(axis=1)
        diff_s = np.convolve(diff, np.ones(5) / 5.0, mode="same")

        r_bot = self.y_bottom - y_top
        base_diff = diff_s[max(0, r_bot - 15) : r_bot].mean()
        if base_diff < 8.0:
            return 0

        r_clamp_bot = 265 - y_top
        below_px = 0
        # Threshold 18.0 eliminates ambient glass reflections in lower curved base
        for r in range(r_bot - 1, r_clamp_bot - 1, -1):
            if diff_s[r] > 18.0:
                below_px += 1
            elif below_px > 10 and diff_s[r] < 12.0:
                break

        # Check physical continuity immediately above clamp (y in [230, 238], r in [70, 78])
        r_above_clamp_start = 230 - y_top
        r_above_clamp_end = 238 - y_top
        above_clamp_mean = diff_s[r_above_clamp_start:r_above_clamp_end].mean()
        crossed_clamp = (above_clamp_mean > 16.0 and (below_px >= 80 or base_diff > 15.0))

        if not crossed_clamp:
            return below_px

        full_below_px = 145
        clamp_px = 23
        above_px = 0

        # Scan upwards above clamp towards meniscus (up to r = 15, y = 175 for 80 mL fills)
        for r in range(r_above_clamp_end, 15, -1):
            if diff_s[r] < 8.0:
                break
            above_px += 1

        return full_below_px + clamp_px + above_px

    def raw_volume_regression_ml(self, px: int) -> float:
        """Evaluates camera perspective regression model for continuous volume in mL."""
        if px < 20:
            return 0.0
        val = float(np.polyval(self.PERSPECTIVE_REGRESSION_COEFFS, px))
        return max(0.0, val)

    def estimate_cumulative_volume_ml(self, px: int) -> int:
        """Estimates cumulative volume (mL) rounded to the nearest 10mL via half-up rounding.
        
        Calibrated across steady gaps between pours on Station 3.
        """
        if px < 20:
            return 0
        raw_ml = self.raw_volume_regression_ml(px)
        return int(10.0 * np.floor((raw_ml + 5.0) / 10.0))
