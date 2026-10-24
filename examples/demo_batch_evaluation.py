"""Batch Quality Inspection: Evaluates all 30 vials in Video 5 to benchmark detection accuracy."""

import csv
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.config import LABELS_DIR, IMAGES_DIR, OUTPUT_DIR, RECIPES
from src.diagnostics.root_cause import RootCauseEngine


def run_batch_evaluation():
    print("=" * 80)
    print("BATCH QUALITY INSPECTION: VIDEO 5 GROUND-TRUTH BENCHMARK (30 VIALS)")
    print("=" * 80)

    recipes_csv = LABELS_DIR / "video5_vial_recipes.csv"
    frame_labels_csv = LABELS_DIR / "video5_frame_labels.csv"

    if not recipes_csv.exists() or not frame_labels_csv.exists():
        print("Missing Video 5 label CSVs!")
        return

    # Load recipe metadata
    vials = []
    with open(recipes_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vials.append({
                "vial": int(row["vial"]),
                "recipe_id": int(row["recipe_id"]),
                "total_ml": float(row["total_ml"]),
                "end_s": float(row["end_s"]),
                "type": row["type"],
                "color_purpose": row.get("color_purpose", ""),
            })

    # Load frame labels
    with open(frame_labels_csv, "r", encoding="utf-8") as f:
        frame_rows = list(csv.DictReader(f))

    engine = RootCauseEngine()
    results = []

    print(f"{'Vial':<5} | {'Rec':<4} | {'Batch Pos':<9} | {'Target':<11} | {'Obs Color':<10} | {'Vol (mL)':<8} | {'Verdict':<6} | {'Defect / Root Cause'}")
    print("-" * 80)

    for i, vinfo in enumerate(vials):
        vid = vinfo["vial"]
        rec_id = vinfo["recipe_id"]
        # Preceding recipe is the previous vial's recipe if it differed
        prev_rec_id = vials[i - 1]["recipe_id"] if i > 0 else None
        batch_pos = ((vid - 1) % 3) + 1  # 1 = Transition, 2 = Steady, 3 = Steady

        # Find the frame closest to end_s
        vial_frames = [r for r in frame_rows if r.get("vial") == str(vid) and r.get("in_fill") == "1"]
        if not vial_frames:
            continue
        # Use last frame of the fill
        best_frame_info = vial_frames[-1]
        img_path = IMAGES_DIR / best_frame_info["filename"]

        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        diag = engine.evaluate_vial(
            frame=frame,
            current_recipe_id=rec_id,
            preceding_recipe_id=prev_rec_id,
            vial_id=vid,
            vial_position_in_batch=batch_pos,
        )

        results.append(diag)

        defect_str = diag.defect_type or "In-spec"
        target_name = RECIPES.get(rec_id, {}).get("target_color", "unknown")
        pos_str = f"#{batch_pos} (Trans)" if batch_pos == 1 else f"#{batch_pos} (Clean)"

        print(
            f"{vid:<5} | {rec_id:<4} | {pos_str:<9} | {target_name:<11} | "
            f"{diag.observed_color_label:<10} | {diag.observed_volume_ml:<8.1f} | "
            f"{diag.verdict:<6} | {defect_str}"
        )

    print("-" * 80)
    total_evaluated = len(results)
    passes = sum(1 for r in results if r.verdict == "PASS")
    carryovers = sum(1 for r in results if r.defect_type == "COLOR_CARRYOVER")
    vol_faults = sum(1 for r in results if r.defect_type in ["UNDERFILL", "OVERFILL"])

    print("\n--- [SUMMARY BENCHMARK METRICS] ---")
    print(f"  Total Vials Evaluated:     {total_evaluated}")
    print(f"  Passed Quality Control:    {passes} ({passes / max(1, total_evaluated) * 100:.1f}%)")
    print(f"  Carryover Defects Flagged: {carryovers}")
    print(f"  Volume Defects Flagged:    {vol_faults}")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_evaluation()
