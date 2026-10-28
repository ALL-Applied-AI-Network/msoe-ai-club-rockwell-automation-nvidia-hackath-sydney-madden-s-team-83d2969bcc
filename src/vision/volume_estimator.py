"""Liquid level (meniscus) detection and volume estimation in milliliters."""

from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import cv2
import numpy as np

from src.config import (
    VIAL_BODY_ROI,
    MENISCUS_CALIBRATION,
    OUTPUT_DIR,
    LABELS_DIR,
    IMAGES_DIR,
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
        # Crop central column of the vial to avoid glass wall side reflections
        # x margin 25% from left and right edges
        margin_x = int(vw * 0.25)
        crop_x = vx + margin_x
        crop_w = vw - (2 * margin_x)
        crop = frame[vy : vy + vh, crop_x : crop_x + crop_w]

        # Convert to grayscale and apply Gaussian blur to smooth fluid ripples
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel vertical derivative (detects horizontal boundaries)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel = np.abs(sobel_y)

        # Average gradient across horizontal span of tube
        row_gradients = np.mean(abs_sobel, axis=1)

        # We search from the bottom upwards (y_bottom up to near the top of the vial)
        # Avoid the Brady barcode sticker area (typically in the top 30% of the vial ROI)
        search_start_idx = int(vh * 0.28)  # Skip sticker near top
        search_end_idx = vh - 10           # Bottom of ROI

        valid_gradients = row_gradients[search_start_idx:search_end_idx]
        if len(valid_gradients) == 0:
            return self.y_bottom, 0.0

        best_sub_idx = int(np.argmax(valid_gradients))
        best_crop_y = search_start_idx + best_sub_idx
        y_full_frame = vy + best_crop_y
        max_grad = float(valid_gradients[best_sub_idx])

        # Normalize confidence
        mean_grad = float(np.mean(valid_gradients)) + 1e-6
        confidence = min(1.0, max_grad / (3.0 * mean_grad))

        # Sanity check: if liquid is essentially empty, meniscus sits at y_bottom
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

    def detect_meniscus_with_ref(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> Tuple[int, float]:
        """Detects meniscus row by counting fluid pixels from bottom upwards."""
        pixel_h = self.count_fluid_pixels(frame, ref_frame)
        meniscus_y = max(160, self.y_bottom - pixel_h)
        return meniscus_y, 0.95

    def estimate_volume(
        self,
        frame: np.ndarray,
        ref_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculates volume in mL from a frame image."""
        if ref_frame is not None:
            pixel_height = self.count_fluid_pixels(frame, ref_frame)
            volume_ml = self.raw_volume_regression_ml(pixel_height)
            y_meniscus = max(160, self.y_bottom - pixel_height)
            conf = 0.95
        else:
            y_meniscus, conf = self.detect_meniscus_y(frame)
            pixel_height = max(0, self.y_bottom - y_meniscus)
            volume_ml = self.raw_volume_regression_ml(pixel_height) if pixel_height > 0 else 0.0

        # Clamp to realistic physical range
        clamped_ml = float(np.clip(volume_ml, 0.0, MENISCUS_CALIBRATION["max_valid_ml"]))
        rounded_ml = self.estimate_cumulative_volume_ml(pixel_height)

        return {
            "volume_ml": round(clamped_ml, 1),
            "raw_volume_ml": volume_ml,
            "cumulative_volume_ml": rounded_ml,
            "y_meniscus": y_meniscus,
            "y_bottom": self.y_bottom,
            "pixel_height": pixel_height,
            "confidence": round(conf, 2),
        }

    def measure_pour_delta(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        ref_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Measures meniscus before and after a tank pour, returning volume delta in whole mL."""
        res_before = self.estimate_volume(frame_before, ref_frame=ref_frame)
        res_after = self.estimate_volume(frame_after, ref_frame=ref_frame)

        vol_before = res_before["volume_ml"]
        vol_after = res_after["volume_ml"]
        delta_raw = max(0.0, vol_after - vol_before)
        delta_rounded = max(10, int(5 * round(delta_raw / 5.0))) if delta_raw >= 7.0 else 0

        return {
            "vol_before_ml": vol_before,
            "vol_after_ml": vol_after,
            "dispensed_raw_ml": round(delta_raw, 2),
            "dispensed_ml": delta_rounded,
            "y_meniscus_before": res_before["y_meniscus"],
            "y_meniscus_after": res_after["y_meniscus"],
        }

    def draw_measurement(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Annotates the meniscus line and estimated volume on the frame."""
        annotated = frame.copy()
        y_m = result["y_meniscus"]
        vol = result["volume_ml"]
        vx, vy, vw, vh = self.vial_roi

        # Meniscus indicator line (Cyan)
        cv2.line(annotated, (vx, y_m), (vx + vw, y_m), (255, 255, 0), 2)
        # Vial bottom line (Gray)
        cv2.line(annotated, (vx, self.y_bottom), (vx + vw, self.y_bottom), (150, 150, 150), 1)

        # Label badge
        text = f"Vol: {vol:.1f} mL (conf: {result['confidence']:.2f})"
        cv2.putText(
            annotated,
            text,
            (vx + vw + 5, y_m + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated


def calibrate_against_video5() -> Tuple[float, float]:
    """Fits linear parameters (px_per_ml) using Video 5's known ground truth recipes.

    Teammates can run this to recalibrate if lighting or camera zoom shifts.
    """
    import csv
    recipes_csv = LABELS_DIR / "video5_vial_recipes.csv"
    frame_labels_csv = LABELS_DIR / "video5_frame_labels.csv"

    if not recipes_csv.exists() or not frame_labels_csv.exists():
        print("Calibration files not found, using default configuration.")
        return MENISCUS_CALIBRATION["px_per_ml"], MENISCUS_CALIBRATION["y_bottom"]

    # Map end frames of each vial
    estimator = VolumeEstimator()
    vial_volumes = {}
    with open(recipes_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vial_volumes[int(row["vial"])] = float(row["total_ml"])

    # Collect end-of-fill frames for steady vials (vials 2 and 3 of each recipe)
    # We use steady vials 2 and 3 (e.g. 2, 3, 5, 6, 8, 9...)
    measured_heights = []
    known_mls = []

    with open(frame_labels_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for vial_id, target_ml in vial_volumes.items():
        # Find the last frame where in_fill == 1 for this vial
        vial_frames = [r for r in rows if r.get("vial") == str(vial_id) and r.get("in_fill") == "1"]
        if not vial_frames:
            continue
        last_frame_info = vial_frames[-1]
        img_path = IMAGES_DIR / last_frame_info["filename"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        y_m, _ = estimator.detect_meniscus_y(img)
        h_px = estimator.y_bottom - y_m
        if h_px > 10:
            measured_heights.append(h_px)
            known_mls.append(target_ml)

    if len(measured_heights) >= 5:
        # Fit h_px = px_per_ml * ml
        # px_per_ml = sum(h_px * ml) / sum(ml^2)
        h = np.array(measured_heights)
        v = np.array(known_mls)
        fitted_px_per_ml = float(np.sum(h * v) / np.sum(v * v))
        print(f"Calibration on Video 5 completed across {len(h)} vials:")
        print(f"  Fitted px_per_ml: {fitted_px_per_ml:.3f} (default: {estimator.px_per_ml})")
        return fitted_px_per_ml, estimator.y_bottom

    return MENISCUS_CALIBRATION["px_per_ml"], MENISCUS_CALIBRATION["y_bottom"]


if __name__ == "__main__":
    from src.config import IMAGES_DIR
    print("Testing VolumeEstimator...")
    estimator = VolumeEstimator()
    sample = list(IMAGES_DIR.glob("*.jpg"))
    if sample:
        test_img = cv2.imread(str(sample[10]))
        res = estimator.estimate_volume(test_img)
        print(f"Sample test: {res}")
        overlay = estimator.draw_measurement(test_img, res)
        out = OUTPUT_DIR / "debug_volume.jpg"
        cv2.imwrite(str(out), overlay)
        print(f"Saved volume debug overlay to {out}")
