"""Root-Cause Diagnostic Engine: evaluates vials and explains faults based on recipe history."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import cv2
import numpy as np

from src.config import RECIPES
from src.models.composition_model import CompositionModel


@dataclass
class DiagnosticResult:
    vial_id: Optional[int]
    recipe_id: int
    verdict: str  # "PASS" or "FAIL"
    defect_type: Optional[str]  # None, "COLOR_CARRYOVER", "UNDERFILL", "OVERFILL", "RECIPE_MISMATCH"
    observed_volume_ml: float
    target_volume_ml: float
    observed_color_label: str
    target_color_label: str
    pred_composition: Dict[str, float]
    target_composition: Dict[str, float]
    composition_delta: Dict[str, float]
    root_cause_explanation: str
    operator_action: str
    next_vial_prediction: str


class RootCauseEngine:
    """Diagnoses vial defects and links anomalies to recipe history and carryover."""

    def __init__(self, composition_model: Optional[CompositionModel] = None):
        self.model = composition_model or CompositionModel()

    def evaluate_vial(
        self,
        frame: np.ndarray,
        current_recipe_id: int,
        preceding_recipe_id: Optional[int] = None,
        vial_id: Optional[int] = None,
        vial_position_in_batch: int = 1,
    ) -> DiagnosticResult:
        """Evaluates a completed vial fill and diagnoses root cause if defective."""
        current_rec = RECIPES.get(current_recipe_id, {})
        target_t100 = float(current_rec.get("t100_ml", 0.0))
        target_t200 = float(current_rec.get("t200_ml", 0.0))
        target_t400 = float(current_rec.get("t400_ml", 0.0))
        target_vol = float(current_rec.get("total_ml", target_t100 + target_t200 + target_t400))
        target_color = current_rec.get("target_color", "unknown")

        # Run composition prediction
        prediction = self.model.predict(frame)
        obs_t100 = prediction["pred_t100_ml"]
        obs_t200 = prediction["pred_t200_ml"]
        obs_t400 = prediction["pred_t400_ml"]
        obs_vol = prediction["pred_total_ml"]
        obs_color = prediction["color_info"]["color_label"]

        delta_t100 = round(obs_t100 - target_t100, 1)
        delta_t200 = round(obs_t200 - target_t200, 1)
        delta_t400 = round(obs_t400 - target_t400, 1)
        vol_error = abs(obs_vol - target_vol)

        # Defect Detection Logic
        defect_type = None
        verdict = "PASS"
        explanation = f"Vial meets Recipe {current_recipe_id} specifications."
        action = "ACCEPT - PROCEED TO PACKAGING"
        next_pred = "Next vial expected PASS."

        # Check Volume
        if vol_error > 6.0:
            verdict = "FAIL"
            defect_type = "OVERFILL" if obs_vol > target_vol else "UNDERFILL"
            explanation = (
                f"Volume error of {vol_error:.1f} mL. Target was {target_vol:.1f} mL, "
                f"but detected {obs_vol:.1f} mL. Check pump flow rates and liquid level sensor."
            )
            action = "QUARANTINE VIAL - CALIBRATE DISPENSE PUMPS"

        # Check Color & Carryover
        # Detect unwanted dye presence
        unwanted_orange = (target_t200 == 0.0 and delta_t200 > 1.5)
        unwanted_blue = (target_t400 == 0.0 and delta_t400 > 1.5)
        color_mismatch = (obs_color != "clear" and target_color != "unknown" and obs_color not in target_color)

        if unwanted_orange or unwanted_blue or (color_mismatch and "green" in obs_color and "blue" in target_color):
            verdict = "FAIL"
            defect_type = "COLOR_CARRYOVER"

            # Cross-reference preceding recipe
            prev_rec = RECIPES.get(preceding_recipe_id, {}) if preceding_recipe_id else {}
            prev_t200 = float(prev_rec.get("t200_ml", 0.0))
            prev_t400 = float(prev_rec.get("t400_ml", 0.0))

            if unwanted_orange and prev_t200 > 0.0:
                explanation = (
                    f"Color Carryover Defect: Detected {delta_t200:.1f} mL excess T-200 (Orange). "
                    f"Preceding Recipe {preceding_recipe_id} dispensed T-200 orange. "
                    f"Residual orange in the shared manifold mixed with current blue fill, producing an out-of-spec {obs_color} tint."
                )
            elif unwanted_blue and prev_t400 > 0.0:
                explanation = (
                    f"Color Carryover Defect: Detected {delta_t400:.1f} mL excess T-400 (Blue). "
                    f"Preceding Recipe {preceding_recipe_id} dispensed T-400 blue. "
                    f"Residual blue dye in nozzle contaminated the new recipe."
                )
            else:
                explanation = (
                    f"Color Anomaly: Observed color '{obs_color}' does not match expected '{target_color}'. "
                    f"Likely caused by manifold carryover or dye ratio variance."
                )

            action = "REJECT / QUARANTINE VIAL"
            next_pred = "Self-flush underway: subsequent vials (2 & 3) predicted to achieve in-spec color."

        return DiagnosticResult(
            vial_id=vial_id,
            recipe_id=current_recipe_id,
            verdict=verdict,
            defect_type=defect_type,
            observed_volume_ml=obs_vol,
            target_volume_ml=target_vol,
            observed_color_label=obs_color,
            target_color_label=target_color,
            pred_composition={"t100_ml": obs_t100, "t200_ml": obs_t200, "t400_ml": obs_t400},
            target_composition={"t100_ml": target_t100, "t200_ml": target_t200, "t400_ml": target_t400},
            composition_delta={"t100_ml": delta_t100, "t200_ml": delta_t200, "t400_ml": delta_t400},
            root_cause_explanation=explanation,
            operator_action=action,
            next_vial_prediction=next_pred,
        )

    def draw_diagnostic_overlay(
        self,
        frame: np.ndarray,
        result: DiagnosticResult,
    ) -> np.ndarray:
        """Draws an industrial quality-control dashboard banner on the image."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Top Banner
        is_pass = (result.verdict == "PASS")
        banner_color = (40, 140, 40) if is_pass else (30, 30, 180)  # Dark green vs Red
        cv2.rectangle(annotated, (0, 0), (w, 55), banner_color, -1)

        # Verdict Header
        vial_str = f"Vial {result.vial_id} | " if result.vial_id else ""
        header = f"{vial_str}Recipe {result.recipe_id} - {result.verdict}"
        if result.defect_type:
            header += f" ({result.defect_type})"

        cv2.putText(
            annotated,
            header,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        sub_text = (
            f"Vol: {result.observed_volume_ml:.1f}/{result.target_volume_ml:.1f}mL | "
            f"Obs Color: {result.observed_color_label.upper()} (Target: {result.target_color_label}) | "
            f"T200 delta: {result.composition_delta['t200_ml']:+.1f}mL"
        )
        cv2.putText(
            annotated,
            sub_text,
            (15, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # Bottom Explanation Bar if defect detected
        if not is_pass:
            cv2.rectangle(annotated, (0, h - 50), (w, h), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                f"ROOT CAUSE: {result.root_cause_explanation[:95]}...",
                (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"ACTION: {result.operator_action} | {result.next_vial_prediction}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (100, 255, 100),
                1,
                cv2.LINE_AA,
            )

        return annotated
