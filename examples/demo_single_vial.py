"""1-Click Demo: Evaluates a single vial frame, diagnoses defects, and renders an annotated image."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.config import IMAGES_DIR, OUTPUT_DIR, RECIPES
from src.vision.roi import draw_rois
from src.vision.volume_estimator import VolumeEstimator
from src.vision.color_extractor import ColorExtractor
from src.diagnostics.root_cause import RootCauseEngine


def run_demo(
    image_filename: str = "vial_fill_20260923_195217_231150.jpg",  # Vial 7 (Recipe 7 with Recipe 6 carryover)
    current_recipe_id: int = 7,
    preceding_recipe_id: int = 6,
    vial_id: int = 7,
):
    print("=" * 70)
    print("FLUID MIXING STATION 3: PHYSICAL AI QUALITY INSPECTION DEMO")
    print("=" * 70)

    img_path = IMAGES_DIR / image_filename
    if not img_path.exists():
        # Fallback to any available image
        all_imgs = list(IMAGES_DIR.glob("*.jpg"))
        if not all_imgs:
            print("No images found in personal_data/images!")
            return
        img_path = all_imgs[0]
        print(f"Specified frame not found, using: {img_path.name}")
    else:
        print(f"Inspecting frame: {img_path.name} (Vial {vial_id}, Recipe {current_recipe_id})")

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Failed to read image at {img_path}")
        return

    # 1. Computer Vision Feature Extraction
    vol_estimator = VolumeEstimator()
    vol_result = vol_estimator.estimate_volume(frame)

    color_extractor = ColorExtractor()
    color_result = color_extractor.extract_liquid_color(frame, vol_result["y_meniscus"])

    # 2. Root-Cause Diagnostic Engine
    engine = RootCauseEngine()
    diag = engine.evaluate_vial(
        frame=frame,
        current_recipe_id=current_recipe_id,
        preceding_recipe_id=preceding_recipe_id,
        vial_id=vial_id,
    )

    # 3. Print Structured Diagnostic Report
    print("\n--- [VISION SENSING RESULTS] ---")
    print(f"  Detected Volume:      {vol_result['volume_ml']} mL (Meniscus Y: {vol_result['y_meniscus']}px)")
    print(f"  Detected Fluid Color: {color_result['color_label'].upper()} (Hue: {color_result['hue']}, Sat: {color_result['saturation']})")
    print(f"  RGB Chromaticity:     R={color_result['r_norm']:.2f}, G={color_result['g_norm']:.2f}, B={color_result['b_norm']:.2f}")

    print("\n--- [INVERSE RECIPE INFERENCE] ---")
    print(f"  Target Composition:   T100: {diag.target_composition['t100_ml']}mL, T200: {diag.target_composition['t200_ml']}mL, T400: {diag.target_composition['t400_ml']}mL")
    print(f"  Inferred Composition: T100: {diag.pred_composition['t100_ml']}mL, T200: {diag.pred_composition['t200_ml']}mL, T400: {diag.pred_composition['t400_ml']}mL")
    print(f"  Composition Delta:    T100: {diag.composition_delta['t100_ml']:+.1f}mL, T200: {diag.composition_delta['t200_ml']:+.1f}mL, T400: {diag.composition_delta['t400_ml']:+.1f}mL")

    print("\n--- [DIAGNOSTIC & ROOT-CAUSE VERDICT] ---")
    print(f"  VERDICT:              {diag.verdict}")
    if diag.defect_type:
        print(f"  DEFECT TYPE:          {diag.defect_type}")
    print(f"  EXPLANATION:          {diag.root_cause_explanation}")
    print(f"  OPERATOR ACTION:      {diag.operator_action}")
    print(f"  NEXT VIAL PREDICTION: {diag.next_vial_prediction}")

    # 4. Create Beautiful Visualization Overlay
    annotated = draw_rois(frame)
    annotated = vol_estimator.draw_measurement(annotated, vol_result)
    annotated = color_extractor.draw_color_swatch(annotated, color_result)
    annotated = engine.draw_diagnostic_overlay(annotated, diag)

    out_file = OUTPUT_DIR / f"vial_{vial_id}_inspection.jpg"
    cv2.imwrite(str(out_file), annotated)
    print(f"\n[Saved Annotated Inspection Image to]:\n  {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fluid Mixing Station 3 - Single Vial Inspection Demo")
    parser.add_argument("--image", type=str, default="vial_fill_20260923_195217_231150.jpg")
    parser.add_argument("--recipe", type=int, default=7)
    parser.add_argument("--prev_recipe", type=int, default=6)
    parser.add_argument("--vial", type=int, default=7)
    args = parser.parse_args()

    run_demo(
        image_filename=args.image,
        current_recipe_id=args.recipe,
        preceding_recipe_id=args.prev_recipe,
        vial_id=args.vial,
    )
