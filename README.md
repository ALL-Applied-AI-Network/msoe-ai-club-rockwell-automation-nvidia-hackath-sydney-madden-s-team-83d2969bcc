# Nozzle Carryover Monitor

A local Streamlit dashboard for browsing the supplied vial-fill images and
Video 5 recipe labels. It groups each fixed-recipe run into three fills, flags
the first vial after a recipe change for carryover inspection, and shows
steady-state comparison fills. It also shows the expected dispense stages using
T-200 → T-400 → T-100, skipping tanks with zero volume. T-300 is never included
in the expected pour.

## Run with Python 3.12

From this folder, create and activate a virtual environment, install the
dependencies, and start the app:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects `image_index.csv`, `images\`, and `labels\` beside `app.py`.
Recipe and per-frame labels are currently available for Video 5; the other
videos can still be browsed by image.
The Recipes & carryover tab includes a recipe-by-color/purpose guide derived
from `labels\video5_vial_recipes.csv`. Any color purpose missing from the
supplied tags is shown as unspecified rather than inferred from an image.

The Video 5 frame labels identify whether a frame is inside a fill and which
vial it belongs to. They do not provide a dispense-stage label or measured
vision-model result. The three-fill comparison therefore displays camera
frames and recipe-history alerts, not automated contamination diagnoses.

## Calibration status

The dashboard does not claim to automatically measure meniscus height, dye
concentration, or contamination. The supplied CSV labels do not include
calibration standards, blank/reference pixel intensities, a vial geometry
calibration, or validated image segmentation. The Calibration tab describes
the data needed to build and validate quadratic volume and Beer-Lambert
optical models before reporting measurements.

The Calibration tab also includes a manual composition-estimate prototype based
on entered volume and color features. It does not process images, load
`ai/model1.joblib`, or provide validated measurements.
