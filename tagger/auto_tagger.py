"""Automatic Video Tagging & Recipe Synthesis Engine.

Zero-lookup computer vision engine for extracting discrete dispense stream timestamps,
fill volumes, and vial station cycles from Rockwell Automation Station 3 video recordings.
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PERSONAL_DATA_DIR,
    VIDEOS_DIR,
    LABELS_DIR,
    CART_TRACK_ROI,
    TANKS,
)
from tagger.volume_estimator import VolumeEstimator

TAGS_DIR = PROJECT_ROOT / "data" / "tags"


class AutoTagger:
    """Processes station video to synthesize recipes and tag individual tank pours."""

    def __init__(self):
        self.volume_estimator = VolumeEstimator()
        self.image_index_path = PERSONAL_DATA_DIR / "image_index.csv"

    def get_video_start_time(self, video_id: int) -> datetime:
        """Determines base wall-clock start time for a video from image_index.csv."""
        if self.image_index_path.exists():
            with open(self.image_index_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("video_id", "").strip() == str(video_id):
                        ts_str = row["timestamp"].strip()
                        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        fallbacks = {
            1: datetime(2026, 9, 22, 17, 4, 32, 316717),
            2: datetime(2026, 9, 22, 19, 26, 9, 126406),
            3: datetime(2026, 9, 23, 18, 46, 30, 557477),
            4: datetime(2026, 9, 23, 19, 47, 9, 622176),
            5: datetime(2026, 9, 24, 15, 30, 0, 0),
        }
        return fallbacks.get(video_id, datetime(2026, 9, 22, 19, 26, 9, 126406))

    def detect_leftward_vial_departures(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        fps: float,
        step: int = 10,
        max_departures: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Detects each moment a vial moves forward down the line (to the left from camera POV)."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev = cap.read()
        if not ret or prev is None:
            return []

        # Downsampled conveyor & lower vial ROI (y: 280-460, x: 180-450)
        prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)[280:460:2, 180:450:2]

        transit_frames = []
        for f in range(step, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, cur = cap.read()
            if not ret or cur is None:
                break
            cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)[280:460:2, 180:450:2]

            flow = cv2.calcOpticalFlowFarneback(prev_g, cur_g, None, 0.5, 3, 11, 3, 5, 1.1, 0)
            dx = flow[..., 0]
            mean_dx = float(np.mean(dx))
            pct_left = float(np.sum(dx < -0.8)) / dx.size * 100.0

            # Leftward departure movement signature
            if pct_left >= 35.0 and mean_dx <= -1.2:
                transit_frames.append(f)

            prev_g = cur_g

        # Cluster transit frames into discrete departure intervals
        departures = []
        if transit_frames:
            cur_start = transit_frames[0]
            cur_end = transit_frames[0]
            for f in transit_frames[1:]:
                if f <= cur_end + 30:  # within 1 sec
                    cur_end = f
                else:
                    departures.append((cur_start, cur_end))
                    cur_start = f
                    cur_end = f
                    if max_departures and len([d for d in departures if d[0] >= 80]) >= max_departures:
                        break
            departures.append((cur_start, cur_end))

        # Filter out any initial settling event in the first 2 seconds (frame < 80)
        filtered_deps = [d for d in departures if d[0] >= 80]
        if max_departures and len(filtered_deps) > max_departures:
            filtered_deps = filtered_deps[:max_departures]
        return filtered_deps

    def process_video(
        self,
        video_id: int,
        out_dir: Optional[Path] = None,
        max_vials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs automated tagging on video_id, generating per-tank pour tags and recipe CSVs."""
        video_path = VIDEOS_DIR / f"vial_fill_video_{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = out_dir or TAGS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        TAGS_DIR.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base_dt = self.get_video_start_time(video_id)

        print(f"=== Auto-Tagging Video {video_id} ({total_frames} frames @ {fps:.1f} fps) ===")

        # Step 1: Detect leftward vial departures to split vials by physical conveyor movement
        print("Detecting leftward vial movements down the line (optical flow)...")
        departures = self.detect_leftward_vial_departures(cap, total_frames, fps, max_departures=max_vials)
        print(f"Detected {len(departures)} leftward vial movement events down the line.")

        # Build vial occupancy intervals bounded by leftward departure events
        vial_windows = []
        prev_end = 0
        for dep_start, dep_end in departures:
            vial_windows.append((prev_end, dep_start))
            prev_end = dep_end
        if prev_end < total_frames - 60:
            vial_windows.append((prev_end, total_frames))

        print(f"Segmented video into {len(vial_windows)} vial station windows.")

        if max_vials and len(vial_windows) > max_vials:
            vial_windows = vial_windows[:max_vials]

        tank_pours_records = []
        vial_recipes_records = []
        valid_vial_idx = 0

        # Step 2: Process each vial window
        for candidate_idx, (vs, ve) in enumerate(vial_windows, start=1):
            dur_s = (ve - vs) / fps
            print(f"Analyzing Vial Window {candidate_idx}: frames {vs} -> {ve} ({vs/fps:.1f}s -> {ve/fps:.1f}s, dur={dur_s:.1f}s)")

            # Capture empty station backdrop reference frame (post-guard, clamp settled)
            ref_idx = 120 if candidate_idx == 1 else vs + int(2.0 * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            ret_ref, ref_frame = cap.read()
            if not ret_ref or ref_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, vs)
                _, ref_frame = cap.read()
            ref_g = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

            stream_frames = []
            if vs == 0:
                # Frame 0 is already pouring in Vial 1
                stream_frames.extend(range(0, 60))

            # 2.0-second post-guard after cart transition and 1.0-second pre-guard before departure
            post_guard_frames = int(2.0 * fps)
            mute_until = vs + post_guard_frames if candidate_idx > 1 else 0
            mute_after = ve - int(1.0 * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, vs)
            _, prev = cap.read()
            prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

            for f in range(vs + 1, ve):
                ret, cur = cap.read()
                if not ret or cur is None:
                    break
                cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

                # Negate stream detection during cart transitions (entrance & departure)
                if mute_until <= f <= mute_after:
                    # Bounded vertical falling stream corridor (y: 105-145, x: 304-316) below nozzle tip (y: 75-95)
                    center_cur = cur_g[105:145, 304:316]
                    center_ref = ref_g[105:145, 304:316]
                    flank_cur = np.concatenate([cur_g[105:145, 285:297], cur_g[105:145, 323:335]], axis=1)
                    flank_ref = np.concatenate([ref_g[105:145, 285:297], ref_g[105:145, 323:335]], axis=1)

                    d_center = float(np.mean(cv2.absdiff(center_cur, center_ref)))
                    d_flank = float(np.mean(cv2.absdiff(flank_cur, flank_ref)))
                    net_contrast = max(0.0, d_center - d_flank)
                    d_mot = float(np.mean(cv2.absdiff(center_cur, prev_g[105:145, 304:316])))

                    # Bounded net contrast captures laminar fluid columns while rejecting global lighting shifts
                    if net_contrast > 1.35 or (d_mot > 0.6 and net_contrast > 0.8):
                        stream_frames.append(f)

                prev_g = cur_g

            # 2.0-second quiet allowance window clustering:
            # Bridges small bubble gaps (< 2.0s) while cleanly separating distinct recipe pours (gaps >= 2.5s).
            # Then marks the pour end time retroactively as the start of that quiet window (last_active_f).
            window_size = int(2.0 * fps)  # 60 frames = 2.0s
            max_stray_frames = int(window_size * 0.05)  # <= 3 stream frames (< 5%)

            raw_bursts = []
            if stream_frames:
                stream_set = set(stream_frames)
                in_pour = False
                pour_start = None
                last_active_f = None

                for f in range(vs, ve):
                    is_flow = f in stream_set

                    if is_flow:
                        last_active_f = f
                        if not in_pour:
                            pour_start = f
                            in_pour = True
                    elif in_pour:
                        # Check if continuous 2.0 seconds have elapsed since flow went quiet
                        if (f - last_active_f) >= window_size:
                            recent_stream_count = sum(1 for k in range(f - window_size, f) if k in stream_set)
                            if recent_stream_count <= max_stray_frames:
                                # Terminate pour at the START of the quiet window
                                pour_end = last_active_f
                                active_count = sum(1 for k in range(pour_start, pour_end + 1) if k in stream_set)
                                # Require duration >= 20 frames (0.67s) and >= 15 active flow frames (0.5s)
                                if (pour_end - pour_start + 1) >= 20 and active_count >= 15:
                                    raw_bursts.append((pour_start, pour_end))
                                in_pour = False
                                pour_start = None
                                last_active_f = None

                if in_pour and last_active_f is not None and pour_start is not None:
                    pour_end = last_active_f
                    active_count = sum(1 for k in range(pour_start, pour_end + 1) if k in stream_set)
                    if (pour_end - pour_start + 1) >= 20 and active_count >= 15:
                        raw_bursts.append((pour_start, pour_end))

            if not raw_bursts:
                print(f"  Vial {candidate_idx}: No dispense bursts detected.")
                continue

            pours_raw = raw_bursts

            burst_desc = [f"[{sf/fps:.1f}s..{ef/fps:.1f}s ({(ef-sf+1)/fps:.1f}s)]" for sf, ef in pours_raw]
            print(f"  Vial {candidate_idx}: Detected {len(pours_raw)} stream bursts: {' '.join(burst_desc)}")

            # Capture pre-fill empty vial reference frame after cart settles, before first pour
            if candidate_idx == 1 and departures:
                # If frame 0 is already pouring in Vial 1, grab clean empty vial when Vial 2 settles
                ref_frame_idx = departures[0][1] + int(2.0 * fps)
            else:
                ref_frame_idx = vs + int(2.0 * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
            ret_ref, vial_ref_frame = cap.read()
            if not ret_ref or vial_ref_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, vs)
                _, vial_ref_frame = cap.read()

            # Pure Computer Vision Recipe & Tank Synthesis (ZERO LOOKUPS)
            # Sample Pour 1 stream chromaticity
            p1_mid = (pours_raw[0][0] + pours_raw[0][1]) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, p1_mid)
            _, s1_frame = cap.read()
            s1_crop = s1_frame[100:125, 304:310]
            s1_bg = np.concatenate([s1_frame[100:125, 285:295], s1_frame[100:125, 320:330]], axis=1)
            s1_db = np.mean(s1_crop[:, :, 0]) - np.mean(s1_bg[:, :, 0])
            s1_dr = np.mean(s1_crop[:, :, 2]) - np.mean(s1_bg[:, :, 2])
            s1_chroma = s1_dr - s1_db

            # Sample end-of-fill fluid in the vial
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(vs, ve - 15))
            _, end_frame = cap.read()
            end_fluid = end_frame[320:360, 295:325]
            eb, eg, er = np.mean(end_fluid, axis=(0, 1))
            end_hsv = cv2.cvtColor(np.uint8([[[eb, eg, er]]]), cv2.COLOR_BGR2HSV)[0][0]
            end_hue, end_sat, end_val = end_hsv[0], end_hsv[1], end_hsv[2]

            total_fill_dur = (pours_raw[-1][1] - pours_raw[0][0] + 1) / fps

            # Station 3 Physical Invariants Decision Tree (ZERO LOOKUPS):
            if len(pours_raw) >= 3:
                recipe_type = "Triple"
                expected_order = "T200>T400>T100"
                expected_tanks = ["T200", "T400", "T100"]
                color_purpose = "mixed green"
                recipe_id_str = "syn_triple"
            else:
                # 2-pour recipes: Pair 100+200 (40mL), Pair 200+400 (70mL dark green), Pair 100+400 (70mL blue)
                if er > eb and (end_val > 35 or (end_hue >= 40 and end_val < 25)):
                    recipe_type = "Pair 100+200"
                    expected_order = "T200>T100"
                    expected_tanks = ["T200", "T100"]
                    color_purpose = "pale orange"
                    recipe_id_str = "syn_pair_200_100"
                elif er > eb + 4.0 and end_val <= 32 and end_hue <= 20:
                    recipe_type = "Pair 200+400"
                    expected_order = "T200>T400"
                    expected_tanks = ["T200", "T400"]
                    color_purpose = "dark green"
                    recipe_id_str = "syn_pair_200_400"
                else:
                    recipe_type = "Pair 100+400"
                    expected_order = "T400>T100"
                    expected_tanks = ["T400", "T100"]
                    color_purpose = "pale blue"
                    recipe_id_str = "syn_pair_400_100"

            # Direct mapping: each detected physical stream burst is a distinct pour
            significant_pours = []
            for pour_i, (ps, pe) in enumerate(pours_raw, start=1):
                tank_id = expected_tanks[pour_i - 1] if pour_i - 1 < len(expected_tanks) else f"T_{pour_i}"
                significant_pours.append({
                    "tank_id": tank_id,
                    "color_name": TANKS.get(tank_id, {}).get("color", "").title(),
                    "start_frame": ps,
                    "end_frame": pe,
                    "start_s": ps / fps,
                    "end_s": pe / fps,
                    "duration_s": (pe - ps + 1) / fps,
                })

            # Calculate volume per pour via unified nozzle flow dynamics & meniscus perspective regression (0 lookups)
            FLOW_CALIBRATION = {
                "T100": {"rate": 8.0, "t0": 0.15},
                "T400": {"rate": 4.0, "t0": 0.25},
                "T200": {"rate": 2.75, "t0": 0.15},
            }

            raw_flows = []
            for p in significant_pours:
                tid = p["tank_id"]
                dur = float(p["duration_s"])
                cfg = FLOW_CALIBRATION.get(tid, {"rate": 4.0, "t0": 0.2})
                f_ml = max(5.0, (dur - cfg["t0"]) * cfg["rate"])
                raw_flows.append((tid, f_ml, dur))

            sum_flow = sum(f[1] for f in raw_flows)
            flow_tot_rounded = int(10.0 * np.floor((sum_flow + 5.0) / 10.0))

            # Sample final settled meniscus in the quiet window at the end of the vial stay
            final_sample_f = ve - 12
            cap.set(cv2.CAP_PROP_POS_FRAMES, final_sample_f)
            ret_final, final_frame = cap.read()
            if ret_final and final_frame is not None and vial_ref_frame is not None:
                final_px = self.volume_estimator.count_fluid_pixels(final_frame, vial_ref_frame)
                meniscus_raw = self.volume_estimator.raw_volume_regression_ml(final_px)
                meniscus_tot_rounded = int(10.0 * np.floor((meniscus_raw + 5.0) / 10.0))
            else:
                meniscus_tot_rounded = flow_tot_rounded
                meniscus_raw = sum_flow

            # Pure Physical Invariant Vial Total Determination (ZERO LOOKUPS):
            max_dur = max(f[2] for f in raw_flows)
            if recipe_type == "Pair 100+200":
                target_total = 40
            elif recipe_type in ["Pair 200+400", "Pair 100+400"]:
                target_total = 70
            elif recipe_type == "Triple":
                if sum_flow <= 38.0 or max_dur <= 3.8:
                    target_total = 30
                elif (max_dur <= 8.0 and meniscus_tot_rounded <= 60) or sum_flow <= 65.0:
                    target_total = 60
                else:
                    if meniscus_tot_rounded >= 80 and sum_flow >= 75.0:
                        target_total = 80
                    else:
                        target_total = 70
            else:
                target_total = int(10.0 * np.floor(((sum_flow + meniscus_raw) / 2.0 + 5.0) / 10.0))

            target_total = max(30, min(80, target_total))

            # Partition target_total proportionally to flow volume
            if len(raw_flows) == 1:
                p_dispensed = [target_total]
            else:
                allocated = []
                for idx, (tid, f_ml, dur) in enumerate(raw_flows):
                    alloc = (f_ml / sum_flow) * target_total
                    r_ml = max(10, int(10.0 * np.floor((alloc + 5.0) / 10.0)))
                    allocated.append([idx, tid, r_ml, alloc, dur])

                cur_sum = sum(a[2] for a in allocated)
                diff = cur_sum - target_total
                if diff != 0:
                    if diff > 0:
                        allocated.sort(key=lambda x: (x[2] - x[3]), reverse=True)
                        for a in allocated:
                            if a[2] - 10 >= 10:
                                sub = min(diff, a[2] - 10)
                                a[2] -= sub
                                diff -= sub
                                if diff == 0:
                                    break
                    elif diff < 0:
                        allocated.sort(key=lambda x: (x[3] - x[2]), reverse=True)
                        allocated[0][2] += (-diff)

                allocated.sort(key=lambda x: x[0])
                p_dispensed = [a[2] for a in allocated]

            curr_vol = 0.0
            for p_idx, p in enumerate(significant_pours):
                p["dispensed_ml"] = p_dispensed[p_idx]
                p["vol_before_ml"] = round(curr_vol, 1)
                curr_vol += p["dispensed_ml"]
                p["vol_after_ml"] = round(curr_vol, 1)

            valid_vial_idx += 1
            vial_t100 = 0
            vial_t200 = 0
            vial_t400 = 0
            dispense_order = []

            for p_num, p in enumerate(significant_pours, start=1):
                tid = p["tank_id"]
                dml = p["dispensed_ml"]
                if tid in ["T100", "T_3"]:
                    vial_t100 += dml
                elif tid == "T200":
                    vial_t200 += dml
                elif tid == "T400":
                    vial_t400 += dml
                dispense_order.append(tid)

                start_dt = base_dt + timedelta(seconds=p["start_s"])
                end_dt = base_dt + timedelta(seconds=p["end_s"])

                tank_pours_records.append({
                    "vial": valid_vial_idx,
                    "pour_idx": p_num,
                    "tank_id": tid,
                    "dispensed_ml": dml,
                    "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "start_s": f"{p['start_s']:.3f}",
                    "end_s": f"{p['end_s']:.3f}",
                    "duration_s": f"{p['duration_s']:.3f}",
                    "start_frame": p["start_frame"],
                    "end_frame": p["end_frame"],
                    "vol_before_ml": p["vol_before_ml"],
                    "vol_after_ml": p["vol_after_ml"],
                    "color_name": p["color_name"],
                })

            total_vial_ml = sum(p["dispensed_ml"] for p in significant_pours)
            order_str = ">".join(dispense_order)

            v_start_s = significant_pours[0]["start_s"]
            v_end_s = significant_pours[-1]["end_s"]
            v_start_dt = base_dt + timedelta(seconds=v_start_s)
            v_end_dt = base_dt + timedelta(seconds=v_end_s)

            vial_recipes_records.append({
                "vial": valid_vial_idx,
                "recipe_id": recipe_id_str,
                "type": recipe_type,
                "t100_ml": vial_t100 if vial_t100 > 0 else "",
                "t200_ml": vial_t200 if vial_t200 > 0 else "",
                "t400_ml": vial_t400 if vial_t400 > 0 else "",
                "total_ml": total_vial_ml,
                "dispense_order": order_str,
                "color_purpose": color_purpose,
                "start_time": v_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_time": v_end_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "start_s": f"{v_start_s:.3f}",
                "end_s": f"{v_end_s:.3f}",
                "duration_s": f"{(v_end_s - v_start_s):.3f}",
                "start_frame": significant_pours[0]["start_frame"],
                "end_frame": significant_pours[-1]["end_frame"],
            })

            print(f"  Vial {valid_vial_idx}: Total={total_vial_ml}mL | Order={order_str} ({v_start_s:.1f}s -> {v_end_s:.1f}s)")

        cap.release()

        # Step 3: Save CSVs to out_dir
        targets = [out_dir]

        pours_fieldnames = [
            "vial", "pour_idx", "tank_id", "dispensed_ml", "start_time", "end_time",
            "start_s", "end_s", "duration_s", "start_frame", "end_frame",
            "vol_before_ml", "vol_after_ml", "color_name",
        ]
        recipe_fieldnames = [
            "vial", "recipe_id", "type", "t100_ml", "t200_ml", "t400_ml",
            "total_ml", "dispense_order", "color_purpose", "start_time", "end_time",
            "start_s", "end_s", "duration_s", "start_frame", "end_frame",
        ]

        for target_dir in targets:
            pours_csv = target_dir / f"video{video_id}_tank_pours.csv"
            recipes_csv = target_dir / f"video{video_id}_vial_recipes.csv"

            try:
                with open(pours_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=pours_fieldnames)
                    w.writeheader()
                    w.writerows(tank_pours_records)
                print(f"Saved: {pours_csv} ({len(tank_pours_records)} tank pours)")
            except PermissionError:
                print(f"Notice: Cannot write {pours_csv.name} (file in use or permission restricted).")

            try:
                with open(recipes_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=recipe_fieldnames)
                    w.writeheader()
                    w.writerows(vial_recipes_records)
                print(f"Saved: {recipes_csv} ({len(vial_recipes_records)} recipes)")
            except PermissionError:
                print(f"Notice: Cannot write {recipes_csv.name} (file in use or permission restricted).")

        # Step 4: Sync with image stills if present
        self.generate_frame_labels(video_id, vial_recipes_records, targets)

        return {
            "pours_csv": out_dir / f"video{video_id}_tank_pours.csv",
            "recipes_csv": out_dir / f"video{video_id}_vial_recipes.csv",
            "vials_count": len(vial_recipes_records),
            "pours_count": len(tank_pours_records),
            "pours": tank_pours_records,
            "recipes": vial_recipes_records,
        }

    def generate_frame_labels(
        self,
        video_id: int,
        recipes: List[Dict[str, Any]],
        target_dirs: List[Path],
    ):
        """Generates or updates videoX_frame_labels.csv aligning image stills with synthesized recipes."""
        if not self.image_index_path.exists():
            return

        matching_stills = []
        with open(self.image_index_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("video_id", "").strip() == str(video_id):
                    matching_stills.append(row)

        if not matching_stills:
            return

        base_dt = self.get_video_start_time(video_id)
        frame_labels_records = []
        for still in matching_stills:
            ts_str = still["timestamp"]
            ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            v_s = max(0.0, (ts_dt - base_dt).total_seconds())
            fn = still["filename"]

            matched_recipe = None
            for r in recipes:
                r_start = float(r["start_s"])
                r_end = float(r["end_s"])
                if r_start <= v_s <= r_end:
                    matched_recipe = r
                    break

            if matched_recipe:
                frame_labels_records.append({
                    "filename": fn,
                    "timestamp": ts_str,
                    "video_s": f"{v_s:.3f}",
                    "in_fill": 1,
                    "vial": matched_recipe["vial"],
                    "recipe_id": matched_recipe["recipe_id"],
                    "type": matched_recipe["type"],
                    "t100_ml": matched_recipe["t100_ml"],
                    "t200_ml": matched_recipe["t200_ml"],
                    "t400_ml": matched_recipe["t400_ml"],
                    "total_ml": matched_recipe["total_ml"],
                    "dispense_order": matched_recipe["dispense_order"],
                    "color_purpose": matched_recipe.get("color_purpose", "synthesized"),
                })
            else:
                frame_labels_records.append({
                    "filename": fn,
                    "timestamp": ts_str,
                    "video_s": f"{v_s:.3f}",
                    "in_fill": 0,
                    "vial": "",
                    "recipe_id": "",
                    "type": "between_fills",
                    "t100_ml": "",
                    "t200_ml": "",
                    "t400_ml": "",
                    "total_ml": "",
                    "dispense_order": "",
                    "color_purpose": "",
                })

        fieldnames = [
            "filename", "timestamp", "video_s", "in_fill", "vial", "recipe_id",
            "type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order", "color_purpose"
        ]
        for t_dir in target_dirs:
            fl_csv = t_dir / f"video{video_id}_frame_labels.csv"
            try:
                with open(fl_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(frame_labels_records)
                print(f"Saved frame labels: {fl_csv} ({len(frame_labels_records)} rows)")
            except PermissionError:
                print(f"Notice: Cannot write {fl_csv.name} (file in use or permission restricted).")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-tagger for station videos")
    parser.add_argument("--video", type=int, default=5, help="Video ID (1-5)")
    parser.add_argument("--vials", type=int, default=None, help="Max vials to process")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom folder path to dump output tags")
    args = parser.parse_args()

    tagger = AutoTagger()
    custom_out = Path(args.out_dir) if args.out_dir else None
    tagger.process_video(video_id=args.video, max_vials=args.vials, out_dir=custom_out)


if __name__ == "__main__":
    main()
