# CSI Fluid Mixing Station — Two-Model AI Package

## Models

Model 1: Synthetic vial-image ingredient fraction estimation.
Model 2: Forecast eight future volume stages from 12 estimated stages.

Ingredient order: T100, T200, T400.

## Requirements

Python 3.10+ and packages in requirements.txt.

Install:

    python -m pip install -r requirements.txt

## Run

Provide 12 chronological synthetic vial images:

    python inference.py --images frame01.png frame02.png frame03.png frame04.png frame05.png frame06.png frame07.png frame08.png frame09.png frame10.png frame11.png frame12.png --output prediction.json

Each image must be 110 pixels wide and 350 pixels high.

Output is JSON with 12 estimated compositions and
eight future compositions.

## Limitations

- Synthetic proof of concept only.
- No real factory concentration accuracy established.
- Predictions are indexed by cumulative dispensed volume fraction,
  not verified elapsed time.
- The procedure is not calibrated for real camera footage.
- Model files are trusted local artifacts; do not load untrusted joblib files.
