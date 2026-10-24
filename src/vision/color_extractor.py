"""Color extraction and classification for fluid inside the vial cylinder."""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

from src.config import VIAL_BODY_ROI, MENISCUS_CALIBRATION, OUTPUT_DIR


class ColorExtractor:
    """Extracts fluid color metrics (RGB, HSV) from the core liquid column."""

    def __init__(self, vial_roi: Tuple[int, int, int, int] = VIAL_BODY_ROI):
        self.vial_roi = vial_roi
        self.y_bottom = MENISCUS_CALIBRATION["y_bottom"]

    def extract_liquid_color(
        self,
        frame: np.ndarray,
        y_meniscus: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Samples the liquid column below the meniscus.

        Args:
            frame: BGR full frame.
            y_meniscus: Detected meniscus row (defaults to halfway if None).

        Returns:
            Dict containing RGB, HSV, normalized RGB, and primary color label.
        """
        vx, vy, vw, vh = self.vial_roi
        if y_meniscus is None:
            # Default to bottom third if meniscus not provided
            y_sample_top = vy + int(vh * 0.6)
        else:
            # Sample 15 pixels below meniscus down to 10 pixels above vial bottom
            y_sample_top = min(self.y_bottom - 15, y_meniscus + 15)

        y_sample_bottom = max(y_sample_top + 10, self.y_bottom - 10)

        # Inward horizontal margin (30%) to sample pure fluid, avoiding curved glass reflections
        margin_x = int(vw * 0.30)
        x_sample_left = vx + margin_x
        x_sample_right = vx + vw - margin_x

        # Safe crop
        h_img, w_img = frame.shape[:2]
        y1 = max(0, min(h_img - 1, y_sample_top))
        y2 = max(y1 + 1, min(h_img, y_sample_bottom))
        x1 = max(0, min(w_img - 1, x_sample_left))
        x2 = max(x1 + 1, min(w_img, x_sample_right))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return self._empty_result()

        # Compute robust median BGR
        median_bgr = np.median(crop, axis=(0, 1)).astype(float)
        b, g, r = float(median_bgr[0]), float(median_bgr[1]), float(median_bgr[2])

        # Convert crop to HSV
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        median_hsv = np.median(crop_hsv, axis=(0, 1)).astype(float)
        h, s, v = float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])

        # Normalized RGB chromaticity (r_norm + g_norm + b_norm = 1.0)
        total_intensity = r + g + b + 1e-6
        r_norm = r / total_intensity
        g_norm = g / total_intensity
        b_norm = b / total_intensity

        color_label = self.classify_hue(h, s, v)

        return {
            "r": round(r, 1),
            "g": round(g, 1),
            "b": round(b, 1),
            "hue": round(h, 1),
            "saturation": round(s, 1),
            "value": round(v, 1),
            "r_norm": round(r_norm, 3),
            "g_norm": round(g_norm, 3),
            "b_norm": round(b_norm, 3),
            "color_label": color_label,
            "sample_box": (x1, y1, x2 - x1, y2 - y1),
        }

    @staticmethod
    def classify_hue(h: float, s: float, v: float) -> str:
        """Classifies fluid into broad industrial color categories.

        OpenCV Hue range: 0 - 179
        """
        # Low saturation indicates clear / water diluent
        if s < 38:
            return "clear"

        # Orange / Yellow range (T-200 dye)
        if 5 <= h < 32:
            return "orange"

        # Green range (T-200 + T-400 carryover or blend)
        if 32 <= h < 85:
            return "green"

        # Blue / Cyan range (T-400 dye)
        if 85 <= h < 135:
            return "blue"

        # Red / Violet edge
        if h < 5 or h >= 165:
            return "red/amber"

        return "mixed"

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "r": 0.0, "g": 0.0, "b": 0.0,
            "hue": 0.0, "saturation": 0.0, "value": 0.0,
            "r_norm": 0.0, "g_norm": 0.0, "b_norm": 0.0,
            "color_label": "unknown",
            "sample_box": (0, 0, 0, 0),
        }

    def draw_color_swatch(
        self,
        frame: np.ndarray,
        color_info: Dict[str, Any],
    ) -> np.ndarray:
        """Draws the sampled region box and a color swatch badge."""
        annotated = frame.copy()
        sx, sy, sw, sh = color_info["sample_box"]
        if sw > 0 and sh > 0:
            # Draw sample area outline
            cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 1)

            # Draw swatch rectangle in top corner
            r, g, b = int(color_info["r"]), int(color_info["g"]), int(color_info["b"])
            swatch_pt1 = (10, 10)
            swatch_pt2 = (60, 40)
            cv2.rectangle(annotated, swatch_pt1, swatch_pt2, (b, g, r), -1)
            cv2.rectangle(annotated, swatch_pt1, swatch_pt2, (255, 255, 255), 1)

            label = f"{color_info['color_label'].upper()} (H:{color_info['hue']:.0f} S:{color_info['saturation']:.0f})"
            cv2.putText(
                annotated,
                label,
                (70, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated


if __name__ == "__main__":
    from src.config import IMAGES_DIR
    extractor = ColorExtractor()
    sample = list(IMAGES_DIR.glob("*.jpg"))
    if sample:
        img = cv2.imread(str(sample[10]))
        res = extractor.extract_liquid_color(img)
        print("ColorExtractor sample test:")
        print(res)
        out = OUTPUT_DIR / "debug_color.jpg"
        cv2.imwrite(str(out), extractor.draw_color_swatch(img, res))
        print(f"Saved color debug overlay to {out}")
