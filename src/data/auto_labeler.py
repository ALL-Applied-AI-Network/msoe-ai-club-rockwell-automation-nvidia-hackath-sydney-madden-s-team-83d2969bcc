"""Auto-labeler: Uses computer vision to automatically detect fill events and generate label CSVs for Videos 1-4."""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.config import PERSONAL_DATA_DIR, IMAGES_DIR, LABELS_DIR, NOZZLE_ROI
from src.vision.roi import crop_roi
from src.vision.volume_estimator import VolumeEstimator
from src.vision.color_extractor import ColorExtractor


class VideoAutoLabeler:
    """Automatically parses video frames, detects dispense cycles, and generates label CSVs."""

    def __init__(self):
        self.image_index_path = PERSONAL_DATA_DIR / "image_index.csv"
        self.vol_estimator = VolumeEstimator()
        self.color_extractor = ColorExtractor()

    def load_video_frames(self, video_id: int) -> List[Dict[str, Any]]:
        """Loads all image rows for a specific video_id from image_index.csv."""
        if not self.image_index_path.exists():
            raise FileNotFoundError(f"Missing {self.image_index_path}")

        frames = []
        with open(self.image_index_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["video_id"].strip() == str(video_id):
                    # Parse timestamp
                    ts_str = row["timestamp"].strip()
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    frames.append({
                        "filename": row["filename"].strip(),
                        "timestamp": ts_str,
                        "datetime": dt,
                        "video_id": video_id,
                    })

        # Sort chronologically
        frames.sort(key=lambda x: x["datetime"])
        return frames

    @staticmethod
    def detect_nozzle_stream(frame: np.ndarray) -> bool:
        """Detects if fluid is actively streaming from the nozzle tip.

        Uses intensity variance and edge density in NOZZLE_ROI.
        """
        crop = crop_roi(frame, NOZZLE_ROI)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        # When a stream is active, fluid boundaries create high standard deviation (> 28)
        return std_dev > 28.0

    def process_video(self, video_id: int, out_dir: Optional[Path] = None) -> Path:
        """Processes all frames of a video and generates the frame labels CSV."""
        out_dir = out_dir or LABELS_DIR
        out_dir.mkdir(exist_ok=True, parents=True)
        out_csv = out_dir / f"video{video_id}_frame_labels.csv"

        print(f"Auto-labeling Video {video_id}...")
        frames = self.load_video_frames(video_id)
        if not frames:
            print(f"No frames found for video_id {video_id} in image_index.csv")
            return out_csv

        video_start_dt = frames[0]["datetime"]

        # Step 1: Detect nozzle activity per frame
        stream_states = []
        for f in frames:
            img_path = IMAGES_DIR / f["filename"]
            if not img_path.exists():
                stream_states.append(False)
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                stream_states.append(False)
                continue
            is_streaming = self.detect_nozzle_stream(img)
            stream_states.append(is_streaming)

        # Step 2: Group stream states into coherent fill intervals (state machine)
        # Smooth out momentary 1-frame flickers
        smoothed_states = stream_states.copy()
        for i in range(1, len(stream_states) - 1):
            if stream_states[i - 1] and stream_states[i + 1]:
                smoothed_states[i] = True

        # Identify contiguous fill blocks
        fill_blocks = []
        in_block = False
        start_idx = 0
        for i, st in enumerate(smoothed_states):
            if st and not in_block:
                in_block = True
                start_idx = i
            elif not st and in_block:
                in_block = False
                end_idx = i - 1
                # Filter out transient false-positives (< 2 frames)
                if (end_idx - start_idx + 1) >= 2:
                    fill_blocks.append((start_idx, end_idx))

        if in_block and (len(smoothed_states) - 1 - start_idx + 1) >= 2:
            fill_blocks.append((start_idx, len(smoothed_states) - 1))

        # Assign vial IDs to blocks
        vial_assignment = {}
        for vial_num, (s_idx, e_idx) in enumerate(fill_blocks, start=1):
            for idx in range(s_idx, e_idx + 1):
                vial_assignment[idx] = vial_num

        print(f"Detected {len(fill_blocks)} vial dispense events in Video {video_id}.")

        # Step 3: Write out frame labels CSV
        fieldnames = [
            "filename", "timestamp", "video_s", "in_fill",
            "vial", "recipe_id", "type", "t100_ml", "t200_ml",
            "t400_ml", "total_ml", "dispense_order", "color_purpose",
        ]

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, f_info in enumerate(frames):
                video_s = (f_info["datetime"] - video_start_dt).total_seconds()
                vial_num = vial_assignment.get(i)
                in_fill = 1 if vial_num is not None else 0

                row = {
                    "filename": f_info["filename"],
                    "timestamp": f_info["timestamp"],
                    "video_s": f"{video_s:.3f}",
                    "in_fill": str(in_fill),
                    "vial": str(vial_num) if vial_num else "",
                    "recipe_id": "",
                    "type": "filling" if in_fill else "between_fills",
                    "t100_ml": "",
                    "t200_ml": "",
                    "t400_ml": "",
                    "total_ml": "",
                    "dispense_order": "",
                    "color_purpose": "",
                }
                writer.writerow(row)

        print(f"Successfully generated: {out_csv} ({len(frames)} frames)")
        return out_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-labeler for unlabelled videos")
    parser.add_argument("--video", type=int, default=1, help="Video ID (1-4)")
    args = parser.parse_args()

    labeler = VideoAutoLabeler()
    labeler.process_video(video_id=args.video)
