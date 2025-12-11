import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from things_eeg2_dataset.paths import layout
from things_eeg2_dataset.visualization.components import (
    processed_eeg_picker,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--project-dir",
    type=str,
    required=True,
    help="Path to project root.",
)
args, unknown = parser.parse_known_args()
project_dir = Path(args.project_dir)

if "approved_pages" not in st.session_state:
    st.session_state["approved_pages"] = set()


def gatekeeper(page_key: str) -> None:
    """Checks for approval for a specific page, displays warning, and halts."""

    if page_key not in st.session_state["approved_pages"]:
        st.error(
            f"⚠️ **Warning: The '{page_key}' page loads large models into memory.**"
        )
        st.warning(
            "Loading this page will consume significant RAM. Do you want to proceed?"
        )

        if st.button(f"Yes, Load {page_key}"):
            st.session_state["approved_pages"].add(page_key)

        st.stop()


if "selected_image_path" not in st.session_state:
    st.session_state["selected_image_path"] = None


@st.cache_data
def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


# --- Configuration ---

top_c1, top_c2 = st.columns(2)
top_original = top_c1.empty()
top_blurred = top_c2.empty()

PROCESSED_EEG_DATA_EXPLORER_PAGE = "Processed EEG Data Explorer"

st.sidebar.header("Choose page")
page = st.sidebar.selectbox(
    "Page",
    [
        PROCESSED_EEG_DATA_EXPLORER_PAGE,
    ],
)
st.sidebar.header("Parameters")

if page == PROCESSED_EEG_DATA_EXPLORER_PAGE:
    st.title(PROCESSED_EEG_DATA_EXPLORER_PAGE)
    processed_eeg_picker(project_dir)

    image_condition = st.sidebar.number_input(
        "Select image condition", min_value=1, max_value=16540, value=1
    )

    # Load channel names from JSON file
    METADATA_FILE = layout.get_metadata_file(project_dir, subject=1)
    metadata = json.load(METADATA_FILE.open("r"))

    offset_enabled = st.sidebar.checkbox(
        "Enable vertical offset for channels", value=True
    )
    CHANNEL_NAMES = metadata["ch_names"]
    selected_channels = st.sidebar.multiselect(
        "Select channels to view",
        CHANNEL_NAMES,
        default=["Oz"],
    )

    selected_eeg_data_path = st.session_state.get("selected_eeg_data_path", None)
    eeg_sample = np.load(selected_eeg_data_path, mmap_mode="r")

    # Two columns for expanders
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1.expander("Show Channel Placement"):
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/f/fb/EEG_10-10_system_with_additional_information.svg",
            caption="Source: https://upload.wikimedia.org/wikipedia/commons/f/fb/EEG_10-10_system_with_additional_information.svg",
            width="stretch",
        )

    if selected_eeg_data_path:
        with exp_col2.expander("Show EEG Sample Info"):
            st.write(f"Available channels: {CHANNEL_NAMES}")
            with layout.get_version_file(project_dir).open("r") as f:
                info_text = f.read()
            st.text(info_text)

        session_index = int(
            st.session_state["selected_session"][-1]
        )  # transform "ses-0X" to int X

        fig = go.Figure()
        time = np.arange(250)  # assuming each epoch = 250 samples
        offset = 0

        for ch in selected_channels:
            ch_idx = CHANNEL_NAMES.index(ch)

            # extract one channel's data
            data_to_plot = eeg_sample[
                session_index - 1,
                image_condition - 1,
                st.session_state["selected_repetition"] - 1,
                ch_idx,
                :,
            ].reshape(-1)

            # Optional: offset each channel vertically so they don't overlap
            fig.add_trace(
                go.Scatter(x=time, y=data_to_plot + offset, mode="lines", name=ch)
            )
            if offset_enabled:
                offset += np.ptp(data_to_plot) * 1.2  # dynamic spacing

        fig.update_layout(
            title=f"Processed EEG Data (Image condition {image_condition})",
            xaxis_title="Time (ms)",
            yaxis_title="Amplitude (µV, offset)",
            height=600,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")
