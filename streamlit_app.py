import streamlit as st
import importlib
import os
from pathlib import Path

import pandas as pd
import joblib
from flask import Flask


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="API Forecast Dashboard",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent


# ==================================================
# MAKE YOUR WINDOWS PATHS WORK ON STREAMLIT CLOUD
# ==================================================

# Your existing files use paths like:
# A2\\M2_hourly.joblib
#
# Streamlit Cloud runs on Linux.
# This adapter converts them safely without changing
# your existing prediction files.

_original_joblib_load = joblib.load
_original_read_csv = pd.read_csv


def fix_path(path):

    if isinstance(path, (str, os.PathLike)):

        path = str(path).replace("\\", "/")

        path_obj = Path(path)

        if not path_obj.is_absolute():
            return str(ROOT / path_obj)

    return path


def portable_joblib_load(filename, *args, **kwargs):
    return _original_joblib_load(
        fix_path(filename),
        *args,
        **kwargs
    )


def portable_read_csv(filepath_or_buffer, *args, **kwargs):

    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        filepath_or_buffer = fix_path(filepath_or_buffer)

    return _original_read_csv(
        filepath_or_buffer,
        *args,
        **kwargs
    )


# Apply compatibility layer
joblib.load = portable_joblib_load
pd.read_csv = portable_read_csv


# ==================================================
# FLASK CONTEXT
# ==================================================

# Your existing prediction files use jsonify().
# We provide the Flask context they need.

flask_app = Flask(__name__)


# ==================================================
# YOUR EXACT API ORDER
# ==================================================

ALL_APIS = [
    "A2", "A10", "A66", "A69", "A42",
    "A12", "A21", "A47", "A31", "A45",
    "A15", "A59", "A18", "A40", "A7"
]

GRANULARITIES = [
    "hourly",
    "daily",
    "weekly",
    "monthly"
]


# ==================================================
# LOAD YOUR EXISTING PREDICTION MODULE
# ==================================================

@st.cache_resource
def load_prediction_module(api, granularity):

    model_number = api[1:]

    module_name = (
        f"{api}.app_M{model_number}_{granularity}"
    )

    return importlib.import_module(module_name)


# ==================================================
# RUN YOUR EXISTING predict() FUNCTION
# ==================================================

def get_prediction(api, granularity):

    try:

        # Load the ORIGINAL prediction file
        module = load_prediction_module(
            api,
            granularity
        )

        # Run its ORIGINAL predict() function
        with flask_app.app_context():

            result = module.predict()

            # Some files return:
            # jsonify(...)
            #
            # Some return:
            # jsonify(...), 500

            if isinstance(result, tuple):

                response = result[0]
                status_code = result[1]

            else:

                response = result
                status_code = 200

            data = response.get_json()

        if status_code >= 400:

            return {
                "error": data.get(
                    "error",
                    "Prediction failed"
                )
            }

        return data

    except Exception as e:

        return {
            "error": str(e)
        }


# ==================================================
# CSS
# ==================================================

st.markdown("""
<style>

    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1800px;
    }

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: #f1f3f6;
        border-radius: 8px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ==================================================
# DASHBOARD TITLE
# ==================================================

st.markdown(
    '<div class="main-title">API Forecast Dashboard</div>',
    unsafe_allow_html=True
)


# ==================================================
# TOP 5 / TOP 10 / TOP 15
# ==================================================

option = st.selectbox(
    "Select APIs",
    ["Top 5", "Top 10", "Top 15"],
    index=2,
    label_visibility="collapsed"
)


# ==================================================
# SELECT APIs
# ==================================================

top_n = int(option.split()[1])

selected_apis = ALL_APIS[:top_n]


# ==================================================
# DISPLAY DASHBOARD
# ==================================================

for api in selected_apis:

    with st.container(border=True):

        st.subheader(api)

        columns = st.columns(4)

        for column, granularity in zip(
            columns,
            GRANULARITIES
        ):

            with column:

                prediction_data = get_prediction(
                    api,
                    granularity
                )

                period_name = granularity.capitalize()

                if "error" in prediction_data:

                    st.metric(
                        label=period_name,
                        value="Error"
                    )

                    st.caption(
                        prediction_data["error"]
                    )

                else:

                    predicted_value = (
                        prediction_data.get(
                            "predicted_call_count",
                            "N/A"
                        )
                    )

                    model_type = (
                        prediction_data.get(
                            "model_type",
                            ""
                        )
                    )

                    st.metric(
                        label=period_name,
                        value=predicted_value
                    )

                    st.caption(model_type)

    st.write("")