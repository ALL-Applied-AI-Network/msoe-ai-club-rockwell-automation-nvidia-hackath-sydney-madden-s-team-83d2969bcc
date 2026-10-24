with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

split_token = "# Rockwell Automation + NVIDIA Innovation Lab"
parts = content.split(split_token, 1)

new_header = """# Team Project Notes: Fluid Mixing Station 3 (Physical AI & Vision)

> **Challenge Track**: Fluid Mixing Station 3 (CSI Testbed)  
> **Core Mission**: Detect fill volume and color defects caused by recipe changeovers, and deliver automated root-cause diagnostics linking optical faults back to preceding recipe fluid carryover.

---

## ⚡ Quickstart (Run in 10 Seconds)

```powershell
# 1. Install dependencies
py -3.12 -m pip install -r requirements.txt

# 2. Run single-vial inspection demo (Vial 7 carryover defect vs Vial 9 clean)
py -3.12 examples/demo_single_vial.py --image "vial_fill_20260923_195325_659355.jpg" --recipe 7 --prev_recipe 6 --vial 7

# 3. Run full 30-vial ground-truth benchmark
py -3.12 examples/demo_batch_evaluation.py
```

---

## 🧠 Critical Technical Insights (Read Before Coding!)

1. **The Shared Nozzle Carryover Mechanism**:
   - Fluids from all tanks feed through a **single shared nozzle**.
   - When switching recipes, residual fluid from the previous recipe clings to the nozzle and contaminates the **first vial** of the new batch (e.g. Recipe 6 Orange carrying over into Recipe 7 Blue creates an out-of-spec **Green** in Vial 7).
   - **Vials 2 & 3 of every batch are steady-state clean** because the line naturally self-flushes after 1 vial.

2. **The 4 Reagent Tanks**:
   - **T-100**: Clear diluent (water)
   - **T-200**: Orange dye
   - **T-300**: *OUT OF SERVICE* (do not use in any recipe)
   - **T-400**: Blue dye
   - *Dispense Sequence*: Always T-200 -> T-400 -> T-100 (T-100 acts as an end-of-pour rinse).

3. **Camera Angle & Meniscus Perspective**:
   - The camera is angled downwards (~15-20 deg pitch), not perpendicular.
   - The meniscus is a **curved ellipse**, not a flat horizontal line.
   - Pixels-per-mL is **non-linear** due to perspective foreshortening (1 mL at the top has fewer pixels than 1 mL at the bottom). A 2nd-degree polynomial calibration is needed.

4. **Non-Linear Optical Absorption (Beer-Lambert Law)**:
   - Dyes do not mix linearly in RGB/Hue space.
   - Even a tiny trace (~1.5 mL) of Orange dye drastically shifts Blue into Green, masking the blue signature.
   - The model must map optical density / log-absorbance (A = -log10(I/I0)) rather than naive linear RGB to avoid wildly underestimating blue volume.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data ["1. Data Ingestion"]
        A["personal_data/images (*.jpg)"] --> D["Image and Timestamp Loader"]
        B["personal_data/image_index.csv"] --> D
        C["personal_data/labels/video5_vial_recipes.csv"] --> D
    end

    subgraph Vision ["2. Computer Vision Pipeline (src/vision/)"]
        D --> E["ROIExtractor (roi.py)"]
        E -->|Nozzle ROI| F["Nozzle Stream Detector (auto_labeler.py)"]
        E -->|Liquid Core ROI| G["VolumeEstimator (volume_estimator.py)"]
        E -->|Liquid Core ROI| H["ColorExtractor (color_extractor.py)"]
        G -->|Meniscus Height -> mL| J["Vial State Vector"]
        H -->|HSV / Normalized RGB| J
    end

    subgraph Intelligence ["3. Intelligence and Diagnostics (src/models/ & src/diagnostics/)"]
        J --> K["CompositionModel (composition_model.py)"]
        K -->|Inferred T100, T200, T400 mL| L["RootCauseEngine (root_cause.py)"]
        C -->|Preceding Recipe Context| L
        L --> M["Diagnostic Output (Terminal Report + Annotated Image)"]
    end
```

---

## 📋 Team Action Plan & Task Board

- [x] **Working Baseline Established**: End-to-end vision, volume, color, composition, and diagnostic pipeline running on Python 3.12.
- [x] **Verified on Real Data**: Vial 7 (Green Carryover Defect) and Vial 9 (Clean Blue Pass) verified.
- [ ] **Subteam A (Vision)**: Refine `src/vision/volume_estimator.py` with a 2nd-degree polynomial curve to correct for camera pitch angle.
- [ ] **Subteam B (ML/Color)**: Enhance `src/models/composition_model.py` using log-absorbance optical density and train on Isaac Sim digital twin data.
- [ ] **Subteam C (Data & Diagnostics)**: Run `src/data/auto_labeler.py` to auto-label Videos 1-4, and prepare the 5-minute final presentation video.

---

"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(new_header + split_token + parts[1])

print("README.md updated successfully!")
