from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
IMAGE_INDEX = ROOT / "image_index.csv"
FRAME_LABELS = LABEL_DIR / "video5_frame_labels.csv"
VIAL_RECIPES = LABEL_DIR / "video5_vial_recipes.csv"
DISPENSE_ORDER = ("200", "400", "100")

st.set_page_config(
    page_title="Nozzle Carryover Monitor",
    layout="wide",
)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def read_csv_or_stop(path: Path, description: str) -> pd.DataFrame:
    if not path.is_file():
        st.error(f"Could not find {description}: `{path}`")
        st.stop()
    try:
        return load_csv(str(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read {description}: {exc}")
        st.stop()


def tank_amounts(recipe: pd.Series) -> dict[str, float]:
    return {
        tank: float(recipe.get(f"t{tank}_ml", 0) or 0)
        for tank in ("100", "200", "300", "400")
    }


def describe_recipe(recipe: pd.Series) -> str:
    tanks = tank_amounts(recipe)
    parts = [
        f"T-{tank}: {amount:g} mL"
        for tank, amount in tanks.items()
        if amount > 0
    ]
    return " · ".join(parts) if parts else "No tank amounts recorded"


def build_transition_table(recipes: pd.DataFrame) -> pd.DataFrame:
    if recipes.empty:
        return recipes.copy()

    ordered = recipes.copy()
    ordered["vial_number"] = pd.to_numeric(ordered["vial"], errors="coerce")
    ordered = ordered.sort_values("vial_number", kind="stable").reset_index(drop=True)

    transitions: list[str] = []
    previous_recipe: str | None = None
    for _, recipe in ordered.iterrows():
        recipe_id = str(recipe.get("recipe_id", "")).strip()
        if previous_recipe is None:
            transitions.append("Unknown — first recorded vial")
        elif recipe_id != previous_recipe:
            transitions.append(f"Check carryover — previous recipe {previous_recipe}")
        else:
            transitions.append("Steady state")
        previous_recipe = recipe_id

    ordered["carryover_status"] = transitions
    return ordered


def get_expected_pour(recipe: pd.Series) -> list[tuple[str, float]]:
    tanks = tank_amounts(recipe)
    return [
        (tank, tanks[tank])
        for tank in DISPENSE_ORDER
        if tanks[tank] > 0
    ]


def estimate_composition(
    volume_ml: float,
    hue: float,
    saturation: float,
    r_norm: float,
    b_norm: float,
) -> dict[str, float]:
    if hue >= 90.0 or (b_norm > 0.45 and r_norm < 0.15):
        t400 = round(volume_ml * min(0.85, max(0.20, saturation / 255.0 * 0.90)), 1)
        return {
            "t100_ml": round(max(0.0, volume_ml - t400), 1),
            "t200_ml": 0.0,
            "t400_ml": t400,
        }

    if hue <= 32.0 or (r_norm > 0.45 and b_norm < 0.15):
        t200 = round(volume_ml * min(0.85, max(0.20, saturation / 255.0 * 0.85)), 1)
        return {
            "t100_ml": round(max(0.0, volume_ml - t200), 1),
            "t200_ml": t200,
            "t400_ml": 0.0,
        }

    hue_clamped = min(103.0, max(22.0, hue))
    blue_weight = (hue_clamped - 22.0) / (103.0 - 22.0)
    orange_weight = 1.0 - blue_weight
    dye_fraction = min(0.85, max(0.20, saturation / 220.0 * 0.80))
    total_dye_ml = volume_ml * dye_fraction
    t200 = round(total_dye_ml * orange_weight, 1)
    t400 = round(total_dye_ml * blue_weight, 1)
    return {
        "t100_ml": round(max(0.0, volume_ml - t200 - t400), 1),
        "t200_ml": t200,
        "t400_ml": t400,
    }


def load_frame_label_map() -> dict[str, dict[str, str]]:
    if not FRAME_LABELS.is_file():
        return {}
    try:
        with FRAME_LABELS.open(newline="", encoding="utf-8-sig") as file:
            return {
                row["filename"]: row
                for row in csv.DictReader(file)
                if row.get("filename")
            }
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        st.warning(f"Frame labels could not be loaded: {exc}")
        return {}


def render_recipe_panel(recipes: pd.DataFrame) -> None:
    st.subheader("Recipe and carryover monitor")
    if recipes.empty:
        st.info("No recipe records are available for this video.")
        return

    transitions = build_transition_table(recipes)
    recipe_guide = transitions.drop_duplicates(subset=["recipe_id"]).copy()
    recipe_guide["color_purpose"] = recipe_guide["color_purpose"].replace(
        "", "Color purpose not specified"
    )
    recipe_guide["recipe"] = recipe_guide.apply(describe_recipe, axis=1)
    st.markdown("#### Recipe by color / purpose")
    st.dataframe(
        recipe_guide[
            [
                "color_purpose",
                "recipe_id",
                "type",
                "recipe",
                "total_ml",
            ]
        ].rename(
            columns={
                "color_purpose": "Color / purpose",
                "recipe_id": "Recipe",
                "type": "Recipe type",
                "recipe": "Tank amounts",
                "total_ml": "Total (mL)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Color / purpose names come from the supplied recipe tags. Blank descriptions "
        "are shown as unspecified; they are not inferred from the images."
    )

    recipe_choices = transitions["recipe_id"].astype(str).drop_duplicates().tolist()
    selected_recipe = st.selectbox(
        "Filter by recipe",
        ["All recipes", *recipe_choices],
        key="recipe_filter",
    )
    visible = transitions
    if selected_recipe != "All recipes":
        visible = transitions[transitions["recipe_id"].astype(str) == selected_recipe]

    flagged_vial_count = int(
        visible["carryover_status"].str.startswith("Check carryover").sum()
    )
    left, right = st.columns(2)
    left.metric("Vials in view", len(visible))
    right.metric("Recipe changes to inspect", flagged_vial_count)
    if flagged_vial_count:
        st.warning(
            "A recipe change is a carryover inspection point. This is a process "
            "alert, not an image-based contamination diagnosis."
        )

    display_columns = [
        column
        for column in (
            "vial",
            "recipe_id",
            "type",
            "total_ml",
            "carryover_status",
            "color_purpose",
        )
        if column in visible.columns
    ]
    st.dataframe(visible[display_columns], hide_index=True, use_container_width=True)

    vial_options = visible.index.tolist()
    if not vial_options:
        return
    selected_index = st.selectbox(
        "Inspect a vial",
        vial_options,
        format_func=lambda index: (
            f"Vial {transitions.loc[index, 'vial']} — "
            f"Recipe {transitions.loc[index, 'recipe_id']}"
        ),
        key="vial_detail",
    )
    recipe = transitions.loc[selected_index]
    amounts = tank_amounts(recipe)
    if amounts["300"] > 0:
        st.error("T-300 is out of service. This recipe must not be dispensed.")
    st.caption(f"Planned recipe: {describe_recipe(recipe)}")
    st.write("Expected component order (T-200 → T-400 → T-100; zero-volume tanks skipped):")
    pour = get_expected_pour(recipe)
    if pour:
        st.code(" → ".join(f"T-{tank}: {amount:g} mL" for tank, amount in pour))
        cumulative = 0.0
        stage_rows: list[dict[str, Any]] = []
        for tank, amount in pour:
            cumulative += amount
            stage_rows.append(
                {
                    "Stage": f"T-{tank}",
                    "Component poured (mL)": amount,
                    "Expected cumulative volume (mL)": cumulative,
                }
            )
        st.dataframe(pd.DataFrame(stage_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Tank volumes are missing; an expected fill sequence cannot be shown.")


st.title("Shared Nozzle Carryover Monitor")
st.caption(
    "Local image and recipe review · T-300 is disabled · "
    "First vial after a recipe change is the carryover inspection point"
)

if not IMAGE_DIR.is_dir():
    st.error(f"Image folder not found: `{IMAGE_DIR}`")
    st.stop()

image_index = read_csv_or_stop(IMAGE_INDEX, "image index")
required_index_columns = {"filename", "timestamp", "video_id"}
if not required_index_columns.issubset(image_index.columns):
    st.error("image_index.csv must contain filename, timestamp, and video_id columns.")
    st.stop()

image_index["video_id"] = image_index["video_id"].astype(str)
frame_labels = load_frame_label_map()

with st.sidebar:
    st.header("Dataset")
    video_options = ["All videos", *sorted(image_index["video_id"].unique())]
    selected_video = st.selectbox("Video", video_options)
    st.divider()
    st.caption("Tank assignments")
    st.write("T-100 · Clear diluent")
    st.write("T-200 · Orange dye")
    st.write("T-300 · Out of service")
    st.write("T-400 · Blue dye")
    st.divider()
    st.caption("No validated optical or volume calibration is loaded.")

filtered = image_index
if selected_video != "All videos":
    filtered = image_index[image_index["video_id"] == selected_video]
filtered = filtered.reset_index(drop=True)

if filtered.empty:
    st.warning("No frames are indexed for this selection.")
    st.stop()

video5_recipes = pd.DataFrame()
if selected_video in ("All videos", "5") and VIAL_RECIPES.is_file():
    video5_recipes = read_csv_or_stop(VIAL_RECIPES, "Video 5 recipe labels")

overview, recipe_tab, calibration_tab = st.tabs(
    ["Frame browser", "Recipes & carryover", "Calibration"]
)

with overview:
    st.subheader("Indexed frames")
    first, second, third = st.columns(3)
    first.metric("Frames in view", len(filtered))
    second.metric("Videos indexed", image_index["video_id"].nunique())
    third.metric("Video 5 labelled frames", len(frame_labels))

    frame_number = st.slider(
        "Frame",
        min_value=1,
        max_value=len(filtered),
        value=1,
        help="Browse the indexed image frames for the selected video.",
    )
    frame = filtered.iloc[frame_number - 1]
    image_path = IMAGE_DIR / Path(str(frame["filename"])).name
    frame_info = frame_labels.get(str(frame["filename"]), {})

    image_col, details_col = st.columns([1.5, 1])
    with image_col:
        if image_path.is_file():
            st.image(str(image_path), caption=str(frame["filename"]), use_container_width=True)
        else:
            st.error(f"Image file is missing: `{image_path}`")

    with details_col:
        st.markdown("#### Frame details")
        st.write(f"**Video:** {frame['video_id']}")
        st.write(f"**Timestamp:** {frame['timestamp']}")
        if frame_info:
            in_fill = str(frame_info.get("in_fill", "0")) == "1"
            st.write(f"**Phase:** {'Filling' if in_fill else 'Between fills'}")
            if in_fill:
                st.write(f"**Vial:** {frame_info.get('vial') or 'Not labelled'}")
                st.write(f"**Recipe:** {frame_info.get('recipe_id') or 'Not labelled'}")
                if frame_info.get("total_ml"):
                    st.write(f"**Target total:** {frame_info['total_ml']} mL")
                if frame_info.get("color_purpose"):
                    st.write(f"**Purpose:** {frame_info['color_purpose']}")
        elif str(frame["video_id"]) != "5":
            st.info("Per-frame recipe labels are available for Video 5 only.")
        else:
            st.info("No frame label is available for this image.")
        st.caption(
            "Camera frames are shown for review. Meniscus location and dye volumes "
            "are not automatically measured."
        )

with recipe_tab:
    if selected_video in ("All videos", "5"):
        render_recipe_panel(video5_recipes)
    else:
        st.info("Recipe details are currently available for Video 5.")

with calibration_tab:
    st.subheader("Calibration setup")
    st.warning(
        "Calibration is not fitted from the supplied labels. Entering illustrative "
        "values here does not produce a validated volume or contamination result."
    )
    st.markdown(
        """
        - **Perspective / meniscus:** Record vial-center meniscus pixel heights at
          known fill volumes, then fit a quadratic calibration. Use a consistent
          centerline reference because the meniscus is curved in the angled view.
        - **Optical response:** Capture a blank reference and known orange/blue
          standards under the same lighting and camera settings. Model
          `A = -log10(I / I₀)` per channel; RGB or hue mixing alone is not a
          calibrated dye-volume estimate.
        - **In-fill checks:** Compare each labelled frame to the expected cumulative
          pour stage. The first vial after a recipe change is the primary
          carryover-inspection point; later vials are steady-state checks.
        """
    )
    st.subheader("Manual composition estimate prototype")
    st.caption(
        "This applies the supplied heuristic to values you enter. It does not analyze "
        "the displayed image, call an API, or load `model1.joblib`."
    )
    volume_ml = st.number_input(
        "Observed liquid volume (mL)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
    )
    hue, saturation, red, blue = st.columns(4)
    hue_value = hue.slider("Hue (OpenCV 0–179)", 0, 179, 60)
    saturation_value = saturation.slider("Saturation (0–255)", 0, 255, 128)
    red_value = red.slider("Normalized red", 0.0, 1.0, 0.3, 0.01)
    blue_value = blue.slider("Normalized blue", 0.0, 1.0, 0.3, 0.01)
    estimate = estimate_composition(
        volume_ml,
        hue_value,
        saturation_value,
        red_value,
        blue_value,
    )
    st.write("Estimated component volumes from the entered values:")
    st.dataframe(
        pd.DataFrame(
            [
                {"Tank": f"T-{tank}", "Estimated volume (mL)": amount}
                for tank, amount in estimate.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "Prototype only: the estimates are not calibrated or validated and should "
        "not be used as machine instructions or contamination results."
    )
    with st.expander("Dispense and safety rules"):
        st.write("Nominal order: T-200 → T-400 → T-100. Skip tanks with zero recipe volume.")
        st.write("T-100 is the final diluent/rinse. T-300 is out of service and must not be used.")
        st.write(
            "The supplied recipe labels include dispense-order values that may differ "
            "from this stated operating rule; confirm the machine configuration before use."
        )
