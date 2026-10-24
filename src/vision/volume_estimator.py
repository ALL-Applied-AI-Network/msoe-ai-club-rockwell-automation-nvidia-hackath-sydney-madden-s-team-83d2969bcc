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

    def estimate_volume(self, frame: np.ndarray) -> Dict[str, Any]:
        """Calculates volume in mL from a frame image."""
        y_meniscus, conf = self.detect_meniscus_y(frame)
        pixel_height = max(0, self.y_bottom - y_meniscus)
        volume_ml = pixel_height / self.px_per_ml

        # Clamp to realistic physical range
        clamped_ml = float(np.clip(volume_ml, 0.0, MENISCUS_CALIBRATION["max_valid_ml"]))

        return {
            "volume_ml": round(clamped_ml, 1),
            "raw_volume_ml": volume_ml,
            "y_meniscus": y_meniscus,
            "y_bottom": self.y_bottom,
            "pixel_height": pixel_height,
            "confidence": round(conf, 2),
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
