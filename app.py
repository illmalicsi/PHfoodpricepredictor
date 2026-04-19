from flask import Flask, jsonify, render_template, request, session, send_file
import base64
import pickle
import pandas as pd
from pathlib import Path
from io import BytesIO
from datetime import datetime
import os
import tempfile
from urllib import request as urllib_request
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

MODEL_PATH = Path(__file__).resolve().parent / "food_price_model.pkl"
HISTORY_SESSION_KEY = "prediction_history"
MAX_HISTORY_ITEMS = 200
MODEL_LOAD_ERROR = ""
MODEL_SOURCE = ""
MODEL_URL_ENV = "MODEL_URL"


def is_git_lfs_pointer(file_path):
    try:
        with file_path.open("rb") as model_file:
            header = model_file.read(128)
        return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def maybe_download_model_from_url():
    model_url = os.getenv(MODEL_URL_ENV, "").strip()
    if not model_url:
        return None

    parsed_url = urlparse(model_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError(f"{MODEL_URL_ENV} must be an http(s) URL")

    local_temp_path = Path(tempfile.gettempdir()) / "food_price_model.pkl"
    urllib_request.urlretrieve(model_url, local_temp_path)
    return local_temp_path


def initialize_model():
    candidate_paths = []

    try:
        downloaded_path = maybe_download_model_from_url()
        if downloaded_path is not None:
            candidate_paths.append((downloaded_path, f"downloaded via {MODEL_URL_ENV}"))
    except Exception as error:  # pragma: no cover - defensive for runtime env failures
        return None, str(error), ""

    candidate_paths.append((MODEL_PATH, "local file"))

    last_error = ""
    for candidate_path, source_label in candidate_paths:
        if not candidate_path.exists():
            last_error = f"{candidate_path.name} not found ({source_label})"
            continue

        if is_git_lfs_pointer(candidate_path):
            last_error = (
                f"{candidate_path.name} appears to be a Git LFS pointer ({source_label}), "
                "not a real pickle binary"
            )
            continue

        try:
            with candidate_path.open("rb") as model_file:
                loaded_model = pickle.load(model_file)
            return loaded_model, "", source_label
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as error:
            last_error = f"Invalid pickle at {candidate_path.name} ({source_label}): {error}"
        except OSError as error:
            last_error = f"Cannot read {candidate_path.name} ({source_label}): {error}"

    return None, last_error, ""

model, MODEL_LOAD_ERROR, MODEL_SOURCE = initialize_model()

CATEGORY_OPTIONS = [
    "cereals and tubers",
    "meat, fish and eggs",
    "miscellaneous food",
    "oil and fats",
    "pulses and nuts",
    "vegetables and fruits",
]

COMMODITY_OPTIONS = [
    "Anchovies",
    "Bananas (lakatan)",
    "Bananas (latundan)",
    "Bananas (saba)",
    "Beans (green, fresh)",
    "Beans (mung)",
    "Beans (string)",
    "Bitter melon",
    "Bottle gourd",
    "Cabbage",
    "Cabbage (chinese)",
    "Calamansi",
    "Carrots",
    "Chicken",
    "Choko",
    "Coconut",
    "Crab",
    "Eggplants",
    "Eggs",
    "Eggs (duck)",
    "Fish (fresh)",
    "Fish (frigate tuna)",
    "Fish (mackerel, fresh)",
    "Fish (milkfish)",
    "Fish (redbelly yellowtail fusilier)",
    "Fish (roundscad)",
    "Fish (slipmouth)",
    "Fish (threadfin bream)",
    "Fish (tilapia)",
    "Garlic",
    "Garlic (large)",
    "Garlic (small)",
    "Ginger",
    "Groundnuts (shelled)",
    "Groundnuts (unshelled)",
    "Maize (white)",
    "Maize (yellow)",
    "Maize flour (white)",
    "Maize flour (yellow)",
    "Mandarins",
    "Mangoes (carabao)",
    "Mangoes (piko)",
    "Meat (beef)",
    "Meat (beef, chops with bones)",
    "Meat (chicken, whole)",
    "Meat (pork)",
    "Meat (pork, hock)",
    "Meat (pork, with bones)",
    "Meat (pork, with fat)",
    "Oil (cooking)",
    "Onions (red)",
    "Onions (white)",
    "Papaya",
    "Pineapples",
    "Potatoes (Irish)",
    "Rice (milled, superior)",
    "Rice (paddy)",
    "Rice (premium)",
    "Rice (regular, milled)",
    "Rice (special)",
    "Rice (well milled)",
    "Semolina (white)",
    "Semolina (yellow)",
    "Shrimp (endeavor)",
    "Shrimp (tiger)",
    "Squashes",
    "Sugar (brown)",
    "Sugar (white)",
    "Sweet Potato leaves",
    "Sweet potatoes",
    "Taro",
    "Tomatoes",
    "Water spinach",
]

CATEGORY_COMMODITY_MAP = {
    "cereals and tubers": [
        "Maize (white)",
        "Maize (yellow)",
        "Maize flour (white)",
        "Maize flour (yellow)",
        "Potatoes (Irish)",
        "Rice (milled, superior)",
        "Rice (paddy)",
        "Rice (premium)",
        "Rice (regular, milled)",
        "Rice (special)",
        "Rice (well milled)",
        "Semolina (white)",
        "Semolina (yellow)",
        "Sweet potatoes",
        "Taro",
    ],
    "meat, fish and eggs": [
        "Anchovies",
        "Chicken",
        "Crab",
        "Eggs",
        "Eggs (duck)",
        "Fish (fresh)",
        "Fish (frigate tuna)",
        "Fish (mackerel, fresh)",
        "Fish (milkfish)",
        "Fish (redbelly yellowtail fusilier)",
        "Fish (roundscad)",
        "Fish (slipmouth)",
        "Fish (threadfin bream)",
        "Fish (tilapia)",
        "Meat (beef)",
        "Meat (beef, chops with bones)",
        "Meat (chicken, whole)",
        "Meat (pork)",
        "Meat (pork, hock)",
        "Meat (pork, with bones)",
        "Meat (pork, with fat)",
        "Shrimp (endeavor)",
        "Shrimp (tiger)",
    ],
    "miscellaneous food": [
        "Coconut",
        "Sugar (brown)",
        "Sugar (white)",
    ],
    "oil and fats": [
        "Oil (cooking)",
    ],
    "pulses and nuts": [
        "Beans (green, fresh)",
        "Beans (mung)",
        "Beans (string)",
        "Groundnuts (shelled)",
        "Groundnuts (unshelled)",
    ],
    "vegetables and fruits": [
        "Bananas (lakatan)",
        "Bananas (latundan)",
        "Bananas (saba)",
        "Bitter melon",
        "Bottle gourd",
        "Cabbage",
        "Cabbage (chinese)",
        "Calamansi",
        "Carrots",
        "Choko",
        "Garlic",
        "Garlic (large)",
        "Garlic (small)",
        "Ginger",
        "Mandarins",
        "Mangoes (carabao)",
        "Mangoes (piko)",
        "Onions (red)",
        "Onions (white)",
        "Papaya",
        "Pineapples",
        "Squashes",
        "Sweet Potato leaves",
        "Tomatoes",
        "Water spinach",
    ],
}


def get_commodity_options(category):
    if category in CATEGORY_COMMODITY_MAP:
        return CATEGORY_COMMODITY_MAP[category]
    return []

REGION_OPTIONS = [
    "Autonomous region in Muslim Mindanao",
    "Cordillera Administrative region",
    "National Capital region",
    "Region I",
    "Region II",
    "Region III",
    "Region IV-A",
    "Region IV-B",
    "Region IX",
    "Region V",
    "Region VI",
    "Region VII",
    "Region VIII",
    "Region X",
    "Region XI",
    "Region XII",
    "Region XIII",
]

REGION_COORDINATES = {
    "Autonomous region in Muslim Mindanao": (6.0, 121.0),
    "Cordillera Administrative region": (17.4, 121.3),
    "National Capital region": (14.6, 121.0),
    "Region I": (16.0, 120.3),
    "Region II": (17.3, 121.8),
    "Region III": (15.4, 120.8),
    "Region IV-A": (14.1, 121.2),
    "Region IV-B": (12.3, 118.5),
    "Region IX": (8.5, 122.8),
    "Region V": (13.3, 123.5),
    "Region VI": (10.7, 122.5),
    "Region VII": (10.3, 123.9),
    "Region VIII": (11.0, 125.0),
    "Region X": (8.4, 124.6),
    "Region XI": (7.3, 126.1),
    "Region XII": (6.5, 124.8),
    "Region XIII": (8.8, 125.7),
}

REGION_MARKET_MAP = {
    "Autonomous region in Muslim Mindanao": "Basilan",
    "Cordillera Administrative region": "Baguio City",
    "National Capital region": "Metro Manila",
    "Region I": "La Union (Ilocos Region)",
    "Region II": "Isabela",
    "Region III": "Pampanga",
    "Region IV-A": "Laguna",
    "Region IV-B": "Palawan",
    "Region IX": "Zamboanga City",
    "Region V": "Albay",
    "Region VI": "Iloilo City",
    "Region VII": "Cebu City",
    "Region VIII": "Tacloban City",
    "Region X": "Cagayan de Oro City",
    "Region XI": "Davao City",
    "Region XII": "Cotabato City",
    "Region XIII": "Butuan City",
}

MONTH_OPTIONS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

YEAR_OPTIONS = list(range(2015, 2036))


def get_month_name(month_number):
    try:
        month_value = int(month_number)
    except (TypeError, ValueError):
        return ""

    if 1 <= month_value <= 12:
        return MONTH_OPTIONS[month_value - 1]
    return ""


def render_home(**context):
    selected_category = context.get("selected_category", "")
    context.setdefault("prediction_history", get_prediction_history())
    return render_template(
        "index.html",
        category_options=CATEGORY_OPTIONS,
        commodity_options=get_commodity_options(selected_category),
        category_commodity_map=CATEGORY_COMMODITY_MAP,
        region_options=REGION_OPTIONS,
        month_options=MONTH_OPTIONS,
        year_options=YEAR_OPTIONS,
        **context,
    )


def get_prediction_history():
    history = session.get(HISTORY_SESSION_KEY, [])
    if isinstance(history, list):
        return history
    return []


def append_prediction_history(entry):
    history = get_prediction_history()
    history.append(entry)
    session[HISTORY_SESSION_KEY] = history[-MAX_HISTORY_ITEMS:]
    session.modified = True


def parse_int_field(form_data, field_name):
    raw_value = form_data.get(field_name, "")
    if raw_value is None:
        raw_value = ""

    raw_value = str(raw_value).strip()
    if not raw_value:
        raise ValueError(f"{field_name.capitalize()} is required")
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{field_name.capitalize()} must be a valid number") from error


def get_required_field(form_data, field_name):
    value = form_data.get(field_name, "").strip()
    if not value:
        raise ValueError(f"{field_name.capitalize()} is required")
    return value


def build_prediction_input(category, commodity, region, year, month):
    latitude, longitude = REGION_COORDINATES[region]
    market = REGION_MARKET_MAP[region]
    return pd.DataFrame([
        {
            "admin1": region,
            "category": category,
            "commodity": commodity,
            "market": market,
            "latitude": latitude,
            "longitude": longitude,
            "year": year,
            "month": month,
        }
    ])


def predict_price_for_inputs(category, commodity, region, year, month):
    if model is None:
        if MODEL_LOAD_ERROR:
            raise ValueError(
                f"Model file is unavailable or invalid ({MODEL_LOAD_ERROR}). "
                f"Ensure food_price_model.pkl is available or set {MODEL_URL_ENV} to a public/direct model URL."
            )
        raise ValueError("Model file not found. Train the model first using train_model.py")

    input_data = build_prediction_input(category, commodity, region, year, month)
    return round(model.predict(input_data)[0], 2)


def get_latest_comparison_price(history, category, commodity, region):
    for entry in reversed(history):
        if (
            entry.get("category") == category
            and entry.get("commodity") == commodity
            and entry.get("region") == region
        ):
            try:
                return float(entry.get("predicted_price", "")), entry
            except (TypeError, ValueError):
                continue
    return None, None


def build_trend_chart_data_uri(history, category, commodity, region, prediction_year=None, prediction_value=None):
    matching_points = []
    for entry in history:
        if (
            entry.get("category") != category
            or entry.get("commodity") != commodity
            or entry.get("region") != region
        ):
            continue

        try:
            year_value = int(entry.get("year"))
            price_value = float(entry.get("predicted_price"))
        except (TypeError, ValueError):
            continue

        matching_points.append((year_value, price_value))

    if prediction_year is not None and prediction_value is not None:
        matching_points.append((int(prediction_year), float(prediction_value)))

    if not matching_points:
        return None

    frame = pd.DataFrame(matching_points, columns=["year", "price"])
    frame = frame.groupby("year", as_index=False)["price"].mean().sort_values("year")

    figure, axis = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    axis.plot(frame["year"], frame["price"], color="#4dd8c8", linewidth=2.4, marker="o", markersize=5)

    if prediction_year is not None and prediction_value is not None:
        axis.scatter([int(prediction_year)], [float(prediction_value)], color="#f0b86a", s=90, zorder=5, marker="*", edgecolors="#1a0e00", linewidths=0.6)
        axis.axvline(int(prediction_year), color="#f0b86a", linestyle="--", linewidth=1, alpha=0.35)

    axis.set_title(f"Price trend for {commodity}", fontsize=12, color="#e8f2fa", pad=10)
    axis.set_xlabel("Year", color="#8fa8be")
    axis.set_ylabel("Price", color="#8fa8be")
    axis.tick_params(axis="x", colors="#8fa8be")
    axis.tick_params(axis="y", colors="#8fa8be")
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.18)
    axis.set_facecolor("#0e1720")
    figure.patch.set_facecolor("#0e1720")

    for spine in axis.spines.values():
        spine.set_color("#2a3a49")

    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


@app.route('/favicon.ico')
@app.route('/favicon.png')
def favicon():
    return "", 204

@app.route('/')
def home():
    return render_home(
        prediction=None,
        error_message=None,
        comparison_previous_price=None,
        comparison_previous_label="",
        comparison_difference=None,
        comparison_difference_display="",
        trend_chart_data_uri=None,
        selected_category="",
        selected_commodity="",
        selected_region="",
        selected_year="",
        selected_month="",
        selected_month_name="",
        prediction_history=get_prediction_history(),
    )

@app.route('/predict', methods=['POST'])
def predict():
    category = request.form.get('category', '')
    commodity = request.form.get('commodity', '')
    region = request.form.get('region', '')
    year = request.form.get('year', '')
    month = request.form.get('month', '')
    prediction_history = get_prediction_history()

    try:
        category = get_required_field(request.form, 'category')
        commodity = get_required_field(request.form, 'commodity')
        region = get_required_field(request.form, 'region')
        year = parse_int_field(request.form, 'year')
        month = parse_int_field(request.form, 'month')

        if region not in REGION_COORDINATES or region not in REGION_MARKET_MAP:
            raise ValueError("Selected region is not supported")

        if category not in CATEGORY_COMMODITY_MAP:
            raise ValueError("Selected category is not supported")

        if commodity not in CATEGORY_COMMODITY_MAP[category]:
            raise ValueError("Selected commodity does not match selected category")

        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")

        prediction_value = predict_price_for_inputs(category, commodity, region, year, month)
        selected_month_name = get_month_name(month)
        previous_price, previous_entry = get_latest_comparison_price(
            prediction_history,
            category,
            commodity,
            region,
        )
        if previous_price is not None:
            comparison_difference = round(prediction_value - previous_price, 2)
            comparison_difference_display = f"{'+' if comparison_difference >= 0 else '-'}₱ {abs(comparison_difference):.2f}"
            comparison_previous_label = f"{previous_entry.get('month', '')} {previous_entry.get('year', '')}".strip()
        else:
            comparison_difference = None
            comparison_difference_display = ""
            comparison_previous_label = ""

        trend_chart_data_uri = build_trend_chart_data_uri(
            prediction_history,
            category,
            commodity,
            region,
            prediction_year=year,
            prediction_value=prediction_value,
        )

        append_prediction_history(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "commodity": commodity,
                "region": region,
                "year": year,
                "month": selected_month_name,
                "predicted_price": prediction_value,
            }
        )
        error_message = None
    except ValueError as error:
        prediction_value = None
        selected_month_name = get_month_name(month)
        error_message = str(error)
        previous_price = None
        comparison_difference = None
        comparison_difference_display = ""
        comparison_previous_label = ""
        trend_chart_data_uri = None

    return render_home(
        prediction=prediction_value,
        error_message=error_message,
        comparison_previous_price=previous_price,
        comparison_previous_label=comparison_previous_label,
        comparison_difference=comparison_difference,
        comparison_difference_display=comparison_difference_display,
        trend_chart_data_uri=trend_chart_data_uri,
        selected_category=category,
        selected_commodity=commodity,
        selected_region=region,
        selected_year=year,
        selected_month=month,
        selected_month_name=selected_month_name,
        prediction_history=get_prediction_history(),
    )


@app.route('/simulate', methods=['POST'])
def simulate():
    payload = request.get_json(silent=True) or request.form

    try:
        category = get_required_field(payload, 'category')
        commodity = get_required_field(payload, 'commodity')
        region = get_required_field(payload, 'region')
        year = parse_int_field(payload, 'year')
        month = parse_int_field(payload, 'month')
        baseline_prediction_raw = payload.get('baseline_prediction', '')
        baseline_prediction = float(baseline_prediction_raw) if str(baseline_prediction_raw).strip() else None

        if region not in REGION_COORDINATES or region not in REGION_MARKET_MAP:
            raise ValueError("Selected region is not supported")

        if category not in CATEGORY_COMMODITY_MAP:
            raise ValueError("Selected category is not supported")

        if commodity not in CATEGORY_COMMODITY_MAP[category]:
            raise ValueError("Selected commodity does not match selected category")

        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")

        simulated_prediction = predict_price_for_inputs(category, commodity, region, year, month)
        delta = None if baseline_prediction is None else round(simulated_prediction - baseline_prediction, 2)

        return jsonify(
            success=True,
            prediction=simulated_prediction,
            month_name=get_month_name(month),
            delta=delta,
            delta_display=(None if delta is None else f"{'+' if delta >= 0 else '-'}₱ {abs(delta):.2f}"),
        )
    except ValueError as error:
        return jsonify(success=False, error=str(error)), 400


@app.route('/export-predictions')
def export_predictions():
    prediction_history = get_prediction_history()
    if not prediction_history:
        return render_home(
            prediction=None,
            error_message="No predictions available to export yet.",
            selected_category="",
            selected_commodity="",
            selected_region="",
            selected_year="",
            selected_month="",
            selected_month_name="",
            trend_chart_data_uri=None,
            prediction_history=[],
        )

    export_data = pd.DataFrame(
        [
            {
                "Timestamp": item.get("timestamp", ""),
                "Category": item.get("category", ""),
                "Commodity": item.get("commodity", ""),
                "Region": item.get("region", ""),
                "Year": item.get("year", ""),
                "Month": item.get("month", ""),
                "Predicted Price": item.get("predicted_price", ""),
            }
            for item in prediction_history
        ]
    )

    csv_bytes = export_data.to_csv(index=False).encode("utf-8-sig")
    filename = f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )

if __name__ == "__main__":
    app.run(debug=True)