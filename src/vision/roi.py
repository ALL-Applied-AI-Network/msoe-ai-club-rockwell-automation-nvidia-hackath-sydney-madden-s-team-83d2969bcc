"""Region of Interest (ROI) management and visualization for vial_fill camera frames."""

from typing import Tuple, Dict, Optional
import cv2
import numpy as np

from src.config import NOZZLE_ROI, VIAL_BODY_ROI, CART_TRACK_ROI, OUTPUT_DIR


def crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop a bounding box [x, y, w, h] from an image."""
    x, y, w, h = roi
    h_img, w_img = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    return image[y1:y2, x1:x2].copy()


def draw_rois(
    image: np.ndarray,
    rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Draw labeled bounding boxes on a copy of the image for visual verification."""
    annotated = image.copy()
    if rois is None:
        rois = {
            "Nozzle ROI": NOZZLE_ROI,
            "Vial Body ROI": VIAL_BODY_ROI,
            "Cart Track ROI": CART_TRACK_ROI,
        }

    default_colors = {
        "Nozzle ROI": (0, 255, 255),    # Yellow
        "Vial Body ROI": (0, 255, 0),    # Green
        "Cart Track ROI": (255, 128, 0), # Light blue
    }
    colors = color_map or default_colors

    for name, (x, y, w, h) in rois.items():
        color = colors.get(name, (0, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        # Background box for label text for legibility
        label_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            annotated,
            (x, y - label_size[1] - 4),
            (x + label_size[0] + 2, y),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            annotated,
            name,
            (x + 1, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


class ROIManager:
    """Helper class to access station ROIs and subregions."""

    def __init__(self):
        self.nozzle = NOZZLE_ROI
        self.vial_body = VIAL_BODY_ROI
        self.cart_track = CART_TRACK_ROI

    def get_nozzle_crop(self, frame: np.ndarray) -> np.ndarray:
        return crop_roi(frame, self.nozzle)

    def get_vial_crop(self, frame: np.ndarray) -> np.ndarray:
        return crop_roi(frame, self.vial_body)

    def get_cart_crop(self, frame: np.ndarray) -> np.ndarray:
        return crop_roi(frame, self.cart_track)


if __name__ == "__main__":
    # Test script: loads a sample frame and saves ROI overlay
    from src.config import IMAGES_DIR
    sample_images = list(IMAGES_DIR.glob("*.jpg"))
    if sample_images:
        test_img_path = sample_images[0]
        img = cv2.imread(str(test_img_path))
        if img is not None:
            overlay = draw_rois(img)
            out_path = OUTPUT_DIR / "debug_rois.jpg"
            cv2.imwrite(str(out_path), overlay)
            print(f"Loaded test frame: {test_img_path.name}")
            print(f"Saved ROI overlay to: {out_path}")
        else:
            print(f"Failed to read image at {test_img_path}")
    else:
        print("No sample images found in personal_data/images")
