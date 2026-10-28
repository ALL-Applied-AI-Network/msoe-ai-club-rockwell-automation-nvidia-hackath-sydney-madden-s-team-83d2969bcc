# Station 3 Automated Video Tagger & Dispense Profiler

A zero-lookup computer vision engine for extracting discrete dispense stream timestamps, fill volumes, and vial station cycles from Rockwell Automation Station 3 video recordings.

---

> [!IMPORTANT]
> ### 🚨 Critical Data Directory Rules (`data/`)
> * **[`data/given_tags/`](file:///C:/Users/granta/ai-club/msoe-ai-club-rockwell-automation-nvidia-hackath-sydney-madden-s-team-83d2969bcc/data/given_tags/)**: Official ground-truth tags provided directly by the hackathon organizers. **CRITICAL: NEVER EDIT, OVERWRITE, OR MODIFY THESE FILES.** They serve as the immutable benchmark reference.
> * **[`data/tags/`](file:///C:/Users/granta/ai-club/msoe-ai-club-rockwell-automation-nvidia-hackath-sydney-madden-s-team-83d2969bcc/data/tags/)**: Synthesized production tags generated automatically by the [`AutoTagger`](file:///C:/Users/granta/ai-club/msoe-ai-club-rockwell-automation-nvidia-hackath-sydney-madden-s-team-83d2969bcc/tagger/auto_tagger.py#L40) engine for Videos 1–5.

---

## 1. Quickstart: Starting the Tagger via CLI

You can start the tagger directly from the terminal or PowerShell using Python 3.12:

```powershell
# Command to run/start the auto-tagger on any video (1-5):
python -m tagger.auto_tagger --video 1 --out-dir data/tags

# Or via the src.data entrypoint:
python src/data/auto_tagger.py --video 1

# Process Video 5 and save to data/tags:
python -m tagger.auto_tagger --video 5 --out-dir data/tags

# Process only a subset (e.g. first 10 vials) for rapid testing:
python -m tagger.auto_tagger --video 5 --vials 10 --out-dir personal_data/eval_video5
```

### CLI Arguments
* `--video` *(int, default: 5)*: Video ID to process (`1`, `2`, `3`, `4`, or `5`).
* `--vials` *(int, optional)*: Maximum number of vials to process (omit to process the entire video).
* `--out-dir` *(str, default: `data/tags`)*: Target directory to write synthesized CSV tags.

---

## 2. Running the Tagger from the Python API at Runtime

The tagger can be invoked programmatically at runtime by other pipelines, test suites, or live inspection nodes:

### A. Full Video Processing via API
```python
from pathlib import Path
from tagger import AutoTagger
# Or: from src.data import AutoTagger

# 1. Initialize the tagger engine
tagger = AutoTagger()

# 2. Process video programmatically at runtime
summary = tagger.process_video(
    video_id=1,
    out_dir=Path("data/tags"),
    max_vials=None,  # Or set an integer cap
)

print(f"Tagged {summary['vials_count']} vials and {summary['pours_count']} discrete pours.")

# 3. Access in-memory records directly without reloading disk CSVs
vial_recipes = summary["recipes"]  # List of dicts with vial cycle metadata
tank_pours = summary["pours"]      # List of dicts with per-pour timestamps & volumes
```

### B. Runtime Volume & Meniscus Sensing via API
For downstream inspection scripts (such as [`RootCauseEngine`](file:///C:/Users/granta/ai-club/msoe-ai-club-rockwell-automation-nvidia-hackath-sydney-madden-s-team-83d2969bcc/src/diagnostics/root_cause.py) or live camera frames), use [`VolumeEstimator`](file:///C:/Users/granta/ai-club/msoe-ai-club-rockwell-automation-nvidia-hackath-sydney-madden-s-team-83d2969bcc/tagger/volume_estimator.py#L14):

```python
import cv2
from tagger.volume_estimator import VolumeEstimator

estimator = VolumeEstimator()

# 1. Meniscus Y detection from a single frame (Sobel vertical derivative)
y_meniscus, conf = estimator.detect_meniscus_y(frame)

# 2. Differential fluid pixel counting vs empty pre-pour reference frame
# (Compensates for the 23px carriage clamp obstruction at y in [242, 265])
fluid_pixels = estimator.count_fluid_pixels(frame, ref_frame=clean_empty_frame)

# 3. Continuous volume (mL) via 2nd-degree perspective projection polynomial
# V(px) = a * px^2 + b * px + c
raw_ml = estimator.raw_volume_regression_ml(fluid_pixels)

# 4. Discrete rounded cumulative volume (nearest 10 mL via half-up rounding)
cumulative_ml = estimator.estimate_cumulative_volume_ml(fluid_pixels)

print(f"Liquid Height: {fluid_pixels} px -> Continuous: {raw_ml:.1f} mL -> Cumulative: {cumulative_ml} mL")
```

---

## 3. Output Tag Files Generated

For every processed video `X`, three synchronized CSV files are written to [`data/tags/`](file:///C:/Users/granta/ai-club/msoe-ai-club-rockwell-automation-nvidia-hackath-sydney-madden-s-team-83d2969bcc/data/tags/):

1. **`videoX_vial_recipes.csv`**: One row per vial with overall cycle timing (`start_s`, `end_s`, `duration_s`), classified recipe type (`Triple`, `Pair 100+200`, `Pair 100+400`, `Pair 200+400`), dispense order, and total fill volume.
2. **`videoX_tank_pours.csv`**: One row per discrete stream burst with start/end timestamps, duration, pre-pour volume, dispensed volume delta, and identified reagent tank (`T100`, `T200`, `T400`).
3. **`videoX_frame_labels.csv`**: Per-frame metadata aligning individual video frame stills with active dispensing states (`in_fill: 1` vs `0`).

---

## 4. Benchmark Accuracy & Validation

### Video 5 Benchmark Results (Ground-Truth Verified)
* **Overall Vials 1–30**: **30 / 30 exact matches (100.0%)**
* **Vials 11–30 (Blind Set)**: **20 / 20 exact matches (100.0%)**
* **Discrete Stream Bursts**: **78 / 78 bursts correctly detected** with zero missed flushes and zero false trigger artifacts from conveyor movement.
* **Volume Rounding**: Strictly adheres to arithmetic half-up rounding to the nearest 10 mL:
  $$\text{cumulative\_ml} = 10 \times \left\lfloor \frac{V + 5.0}{10.0} \right\rfloor$$
* **Zero Hardcoding**: All recipe classifications and volume measurements derive strictly from computer vision features, perspective trigonometry, and fluid flow dynamics.

To re-run the benchmark evaluation on Video 5:
```powershell
python scripts/eval_vials_11_to_30.py
```

---

## 5. Technical Architecture & Vision Pipeline

```
Video Stream
     │
     ▼
[Step 1: Conveyor Optical Flow]  ──► Detects leftward cart motion (y: 280-460, x: 180-450)
     │                               Segments discrete vial station windows [vs, ve]
     ▼
[Step 2: 2.0s Post-Guard]        ──► Mutes stream detection during cart arrival & clamp closure
     │
     ▼
[Step 3: Falling Stream Corridor]──► Samples strictly below nozzle tip (y: 105-145)
     │                               Ignores clinging / breathing nozzle bubbles
     ▼
[Step 4: Bounded Backdrop Contrast]► Net contrast = max(0, Center - Flanks) vs empty backdrop
     │                               Detects steady laminar flow; rejects ambient lighting shifts
     ▼
[Step 5: Quiet Allowance Window] ──► 2.0s allowance bridges micro-bubbles; separates distinct pours
     │
     ▼
[Step 6: Meniscus Pixel Counting]──► Counts fluid pixels (2.65 px/mL) relative to pre-fill reference
     │                               Compensates for 23 px carriage clamp bracket (y: 242-265)
     ▼
[Step 7: 2nd-Degree Perspective] ──► V(px) = a * px^2 + b * px + c (polynomial foreshortening)
     │
     ▼
Synchronized CSV Tag Generation (data/tags/)
```

### Key Vision Modules
1. **Conveyor Optical Flow**: Farneback optical flow on the lower track ($y \in [280, 460], x \in [180, 450]$) detects leftward displacement ($\bar{dx} \le -1.2, \ge 35\%$ leftward pixels) to identify the moment an old vial departs and a new cart enters.
2. **Falling Stream Corridor Isolation ($y \in [105, 145]$)**: By sampling strictly below the nozzle orifice ($y \le 95$), clinging bubbles, "breathing" bubbles that expand and suck back into the nozzle, and lingering meniscus wobble are physically excluded from the stream zone.
3. **Spatially Bounded Backdrop-Contrast (Laminar Flow Solution)**:
   $$\text{net\_contrast} = \max(0, d_{\text{center}} - d_{\text{flank}})$$
   * Samples the central vertical fluid column ($x \in [304, 316]$) directly against its flanking background strips ($x \in [285, 297] \cup [323, 335]$) relative to the empty station reference.
   * **Laminar Flow**: A smooth, uniform stream has zero frame-to-frame motion delta, but produces high contrast ($\ge 1.35$) against the backdrop, capturing it as 1 continuous pour.
   * **Lighting Immunity**: Ambient room lights or camera auto-exposure illuminate center and flanks equally, resulting in $\text{net\_contrast} \approx 0$ (zero false triggers).
4. **Quiet Allowance Window ($2.0\text{s}$)**: Bridges brief bubble interruptions ($< 2.0\text{s}$) within an active pour without bridging genuine multi-second pauses between recipe dispenses.
5. **Camera Perspective Trigonometric Regression & Volume Rounding**:
   * Evaluated strictly during the steady gap after each pour settles.
   * **Perspective Trigonometry**: Due to camera tilt and foreshortening relative to the vertical cylinder, a quadratic regression model:
     $$V(px) = a \cdot px^2 + b \cdot px + c$$
     captures the curved hemispherical base ($c$), cylindrical barrel ($b$), and second-order perspective tilt foreshortening ($a$).
   * **Rounding**: Evaluates continuous regression and rounds cumulative volume to the nearest $10\text{ mL}$, matching physical recipe increments.
   * **Physical Clamp Handling**: Compensates for the $23\text{ px}$ clamp bracket obstruction ($y \in [242, 265]$) with strict physical continuity verification immediately above the clamp ($y \in [230, 238]$).
