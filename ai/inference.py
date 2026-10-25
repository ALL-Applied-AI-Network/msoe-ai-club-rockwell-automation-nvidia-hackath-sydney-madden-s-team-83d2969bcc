from pathlib import Path
import argparse
import json

import cv2
import joblib
import numpy as np


HOME = Path(__file__).resolve().parent

MODEL1_PATH = HOME / "models" / "model1.joblib"
MODEL2_PATH = HOME / "models" / "model2.joblib"


def normalize_composition(values):
    """Ensure ingredient fractions are nonnegative and sum to one."""
    array = np.maximum(np.asarray(values, dtype=float), 0)
    totals = array.sum(axis=-1, keepdims=True)
    safe = np.maximum(totals, 1e-12)
    result = array / safe

    zero = totals[..., 0] <= 1e-12
    result[zero] = 1.0 / 3.0
    return result


def extract_features(image_path):
    """Extract nine synthetic-image color features."""
    image = cv2.imread(str(image_path))

    if image is None or image.shape != (350, 110, 3):
        raise ValueError(
            f"Expected a 350x110 synthetic vial image: {image_path}"
        )

    region = image[50:310, 43:68].astype(np.float32)
    distance = np.linalg.norm(region - 75.0, axis=2)
    mask = distance > 25.0

    if mask.sum() < 20:
        # Same synthetic-only fallback used in T5.5.
        pixels = region[254:259, :, :].reshape(-1, 3)
    else:
        pixels = region[mask]

    rgb = pixels[:, ::-1] / 255.0

    median = np.median(rgb, axis=0)
    mean = np.mean(rgb, axis=0)
    chroma = median / max(median.sum(), 1e-8)

    return np.concatenate([median, mean, chroma])


def predict(image_paths, model1, model2):
    """Estimate 12 compositions and predict eight future stages."""
    if len(image_paths) != 12:
        raise ValueError("Exactly 12 chronological images are required.")

    features = np.asarray([
        extract_features(path) for path in image_paths
    ])

    assert features.shape == (12, 9)

    observed = normalize_composition(
        model1.predict(features)
    )

    future = normalize_composition(
        model2.predict(
            observed.reshape(1, 36)
        ).reshape(8, 3)
    )

    return observed, future


def main():
    parser = argparse.ArgumentParser(
        description="CSI two-model synthetic composition inference"
    )
    parser.add_argument(
        "--images",
        nargs=12,
        required=True,
        help="12 chronological synthetic vial image paths"
    )
    parser.add_argument(
        "--output",
        default=str(HOME / "CSI_Two_Model_Demo" / "inference.json")
    )
    args = parser.parse_args()

    for path in (MODEL1_PATH, MODEL2_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model: {path}")

    model1 = joblib.load(MODEL1_PATH)
    model2 = joblib.load(MODEL2_PATH)

    paths = [Path(p) for p in args.images]
    observed, future = predict(paths, model1, model2)

    result = {
        "model1_estimates": observed.tolist(),
        "model2_forecast": future.tolist(),
        "ingredient_order": ["T100", "T200", "T400"],
        "observed_stages": list(range(1, 13)),
        "forecast_stages": list(range(13, 21)),
        "progress_axis": "cumulative_dispensed_volume_fraction",
        "input_domain": "procedural_synthetic_images",
        "real_concentration_validated": False
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("=== T7.2 STANDALONE INFERENCE ===")
    print("Synthetic images processed:", len(paths))
    print("Observed compositions:", len(observed))
    print("Future compositions:", len(future))
    print("Ingredient order: T100, T200, T400")
    print("Output:", output)
    print("Models retrained: NO")
    print("T7.2 status: COMPLETE")


if __name__ == "__main__":
    main()
