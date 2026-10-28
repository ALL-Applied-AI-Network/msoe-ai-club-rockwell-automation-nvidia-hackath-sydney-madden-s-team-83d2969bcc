# Fluid Mixing Station: Fluid Composition Model Training Specification & RFC

**Project**: Rockwell Automation + NVIDIA Innovation Lab (Fall 2026)  
**Track**: Fluid Mixing Station 3 (Physical AI & Vision)  
**Document Purpose**: Engineering Request for Comments (RFC) on training a new fluid composition model to estimate percentage fractions of $[T_{100}, T_{200}, T_{400}]$ across arbitrary vial fill levels ($10\text{–}70\text{ mL}$).  
**Status**: Open for Team Discussion & Decision  

---

## 1. Executive Summary & Objective

Our inspection pipeline needs to accurately determine the fluid breakdown of liquid inside Station 3 vials across fill levels ranging from **$10\text{ mL}$ to $70\text{ mL}$**:
* **$T_{100}$**: Clear diluent (water)
* **$T_{200}$**: Orange dye
* **$T_{400}$**: Blue dye

### The Intensive Property Principle
* **Percentage composition is an intensive property**: A mixture of $20\text{ mL}$ Orange + $20\text{ mL}$ Blue exhibits the identical spectral absorption ratio ($50\%/50\%$) whether it is at $15\text{ mL}$, $40\text{ mL}$, or $70\text{ mL}$ total fill.
* **Volume is an extensive property**: Measured directly in mL by our calibrated 2nd-degree perspective polynomial model ($V_{\text{total}}$).
* **Decoupled Formulation**: The model predicts normalized percentage fractions $[f_{\text{clear}}, f_{\text{orange}}, f_{\text{blue}}]$ summing to $1.0$ ($100\%$). The downstream API simply multiplies:
  $$V_{\text{tank}} = f_{\text{tank}} \times V_{\text{total}}$$
  This eliminates the need for the color model to infer height or volume, drastically reducing model complexity and training variance.

---

## 2. Optical Physics: Beer-Lambert Law vs. Naive RGB

Liquid dyes in transmitted light mix **subtractively according to the Beer-Lambert Law**, not additively like RGB computer screens:
$$I(\lambda) = I_0(\lambda) \cdot 10^{-\sum_{i} \epsilon_i(\lambda) \cdot c_i \cdot l}$$

Where:
* $I_0(\lambda)$ is incoming backlight intensity.
* $I(\lambda)$ is transmitted light reaching the camera.
* $c_i$ is the concentration of dye $i$ ($T_{200}$ Orange, $T_{400}$ Blue).
* $l$ is optical path length (the vial's inner diameter, constant at $\approx 25\text{ mm}$).

### Why Naive RGB Fails
In linear RGB space, color blending is highly non-linear. Even $1.5\text{ mL}$ of Orange dye strongly absorbs blue light, shifting the perceived hue from blue directly into dark green/brown. A model trained on raw RGB struggles with subtle carryover traces.

### Why Optical Absorbance Solves It
By converting transmission to **Optical Absorbance (Optical Density)**:
$$A_c = -\log_{10}\left(\frac{I_c + 1}{I_{0, c} + 1}\right) \quad \text{for } c \in \{R, G, B\}$$

**Absorbances add linearly with dye concentration:**
$$\begin{bmatrix} A_R \\ A_G \\ A_B \end{bmatrix} = c_{\text{orange}} \begin{bmatrix} a_{R, \text{orange}} \\ a_{G, \text{orange}} \\ a_{B, \text{orange}} \end{bmatrix} + c_{\text{blue}} \begin{bmatrix} a_{R, \text{blue}} \\ a_{G, \text{blue}} \\ a_{B, \text{blue}} \end{bmatrix}$$

Clear diluent ($T_{100}$) scales down dye concentrations $c_{\text{orange}}$ and $c_{\text{blue}}$, proportionally decreasing total absorbance magnitude $\|\mathbf{A}\|$ without altering the spectral absorbance ratio $A_R / A_B$. Training the model in this space transforms non-linear color mixing into a linear / near-linear regression problem.

---

## 3. Data Ingestion & Preprocessing Pipeline

```
Raw Camera Frame (640x480)
           │
           ▼
[Meniscus Detection via VolumeEstimator] ──► Extracts y_meniscus (180 - 380 px)
           │
           ▼
[Dynamic Sub-Meniscus Crop] ───────────────► Central core: x in [305, 325]
           │                                 Vertical: y_bottom - 15 to y_meniscus + 15
           ▼
[Carriage Clamp Exclusion Mask] ───────────► Exclude metal bracket (y in [240, 268])
           │
           ▼
[Backlight Normalization (I_0)] ───────────► Flank sample: x in [280, 290], y in [180, 220]
           │
           ▼
[10-Dimensional Feature Extraction Vector]
```

### 10-Dimensional Feature Vector ($\mathbf{x}$)
1. **Median Absorbance Vector**: $[A_R, A_G, A_B]$
2. **Absorbance Chromaticity Ratios**:
   $$\hat{A}_R = \frac{A_R}{A_R + A_G + A_B + \epsilon}, \quad \hat{A}_B = \frac{A_B}{A_R + A_G + A_B + \epsilon}$$
3. **Total Absorbance Norm**: $\|\mathbf{A}\| = \sqrt{A_R^2 + A_G^2 + A_B^2}$ *(high = concentrated dye; low = heavy diluent)*.
4. **HSV Color Metrics**: Median Hue ($H \in [0, 180]$), Saturation ($S \in [0, 255]$), Value ($V \in [0, 255]$).
5. **Fluid Height Feature**: $h_{\text{fluid}} = y_{\text{bottom}} - y_{\text{meniscus}}$ (to allow minor optical path depth compensation).

---

## 4. Open Architecture Decisions for the Team

The following architectural choices are open for team consensus:

### 🗳️ Decision 1: Model Architecture & Inference Engine

| Option | Architecture | Pros | Cons | Recommended For |
| :--- | :--- | :--- | :--- | :--- |
| **Option A** | **Physics-Guided Constrained NNLS** *(Non-Negative Least Squares)* | • Solves $A = \mathbf{E} \mathbf{c}$ analytically.<br>• Zero risk of overfitting.<br>• Requires only 2 reference calibration vectors. | Cannot learn second-order optical refraction artifacts from glass curve. | Fast baseline, mathematical transparency. |
| **Option B** | **Multi-Layer Perceptron (MLP) with Softmax Output** *(Recommended)* | • Softmax layer mathematically guarantees $f_i \ge 0$ and $\sum f_i = 1.0$.<br>• Captures slight optical non-linearities.<br>• Fast CPU inference ($< 2\text{ ms}$). | Requires 100–300 training samples to tune weights. | Production accuracy, clean percentage outputs. |
| **Option C** | **Gradient Boosted Trees (LightGBM / XGBoost)** | • Highly robust to outliers.<br>• Generates feature importance rankings. | Requires post-hoc normalization step (`preds / np.sum(preds)`). | Tabular data experts, rapid experimentation. |
| **Option D** | **Lightweight End-to-End CNN (MobileNetV3 / Custom 4-Layer)** | • Bypasses manual feature engineering; reads raw pixel crops directly. | • Highest risk of overfitting to real rig lighting.<br>• Heavier deployment footprint. | If spatial fluid patterns (e.g. unmixed layers) matter. |

*Team question: Should we implement Option B (MLP with Softmax) as primary, with Option A (NNLS) as a deterministic fallback?*

---

### 🗳️ Decision 2: Training Dataset Strategy

* **Strategy 1: NVIDIA Isaac Sim Digital Twin Data Only**
  * *Approach*: Generate 2,000 synthetic vial renders across randomized mixture grids and lighting variations.
  * *Pros*: Infinite perfectly labeled ground truth.
  * *Cons*: Real-world domain gap (real LED backlight color temperature and sensor noise differs from digital twin shader).
* **Strategy 2: Physical Line Dataset Only (Videos 1–5 Stills)**
  * *Approach*: Extract settled frames from steady-state Vials 2 & 3 (known ground truth) and intermediate pour pauses (pure single-tank states).
  * *Pros*: 100% matched to real camera optics and station illumination.
  * *Cons*: Limited to recipes present in Videos 1–5 (less dense mixture coverage).
* **Strategy 3: Hybrid Sim-to-Real Transfer (Recommended)**
  * *Approach*: Pre-train the MLP on Isaac Sim synthetic variations, then fine-tune / calibrate the final layer on physical Video 5 ground-truth frames.
  * *Pros*: Combines dense synthetic coverage with real-world optical calibration.

*Team question: Do we have access to render custom Isaac Sim trials now, or should we bootstrap directly on real Video 1–5 stills first?*

---

### 🗳️ Decision 3: Loss Function Formulation

* **Choice A: Kullback-Leibler (KL) Divergence / Softmax Cross-Entropy**:
  $$\mathcal{L} = -\sum_{i \in \{\text{clear}, \text{orange}, \text{blue}\}} f_i \log(\hat{f}_i + \epsilon)$$
  *Treats the percentage composition as a probability simplex; penalizes small probability shifts heavily (good for catching small carryover traces).*
* **Choice B: Mean Absolute Error (MAE) with Dirichlet Constraint**:
  $$\mathcal{L} = \frac{1}{3} \sum |f_i - \hat{f}_i| + \lambda \left| 1.0 - \sum \hat{f}_i \right|$$
  *Directly optimizes error in percentage points.*

*Team question: Do we prioritize catching tiny $< 5\%$ carryover traces (favoring Choice A), or overall mean error (Choice B)?*

---

### 🗳️ Decision 4: Integration Role in the Diagnostic Pipeline

* **Role 1: Dual-Consensus Voting alongside End-of-Stream Tagger**:
  * Active pour tank order is determined by combining the End-of-Stream chromaticity score with Model 1's incremental volume delta $\Delta \vec{V} = \vec{V}_k - \vec{V}_{k-1}$.
* **Role 2: Specialized Mixture & Carryover Analyzer**:
  * End-of-Stream chromaticity handles $100\%$ of stream burst tank identification ($T_{100}, T_{200}, T_{400}$).
  * This new model is called **only on the final settled vial** to estimate $[f_{\text{clear}}, f_{\text{orange}}, f_{\text{blue}}]$ and detect unexpected carryover contamination.

*Team question: Should we keep the stream tagger and this vial composition model completely separated (Role 2) to prevent cascading errors?*

---

## 5. Verification & Acceptance Criteria

Any candidate model must meet three physical test criteria before deployment:

1. **Pure Single-Tank Accuracy ($\le 2\%$ Error)**:
   * Pure Orange ($T_{200}$): Model outputs $[0\%, 100\%, 0\%] \pm 2\%$.
   * Pure Blue ($T_{400}$): Model outputs $[0\%, 0\%, 100\%] \pm 2\%$.
   * Pure Water ($T_{100}$): Model outputs $[100\%, 0\%, 0\%] \pm 3\%$.
2. **Carryover Sensitivity (Trace Detection)**:
   * On a $70\text{ mL}$ Blue recipe with a $5\text{ mL}$ Orange carryover ($7.1\%$ Orange fraction), the model must output $f_{\text{orange}} \ge 5.0\%$, correctly identifying the contaminant.
3. **Volume Invariance Test**:
   * A nominal $50\%/50\%$ mixture tested at $15\text{ mL}$, $35\text{ mL}$, and $65\text{ mL}$ must output identical percentages within $\pm 2\%$.
