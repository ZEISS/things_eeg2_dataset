from pathlib import Path

import streamlit as st
from PIL import Image

from things_eeg2_dataset.paths import layout


def processed_eeg_picker(project_dir: Path) -> None:
    subjects = [
        int(subject.name.split("-")[-1])
        for subject in layout.get_processed_dir(project_dir).iterdir()
    ]
    use_training_data = st.sidebar.toggle("Use training data", value=True)
    selected_subject = st.sidebar.selectbox("Select a subject", sorted(subjects))

    if use_training_data:
        selected_session = st.sidebar.selectbox(
            "Select a session", ["ses-01", "ses-02", "ses-03", "ses-04"]
        )
    else:
        selected_session = st.sidebar.selectbox(
            "Select a session", [f"ses-{i:02d}" for i in range(1, 81)]
        )

    max_reps = 2 if use_training_data else 20
    repetition = st.sidebar.number_input(
        "Select repetition", min_value=1, max_value=max_reps, value=1
    )
    if selected_subject:
        SUBJECT_DATA_DIR_PROCESSED = layout.get_processed_subject_dir(
            project_dir, selected_subject
        )
        if use_training_data:
            eeg_data_file = layout.get_eeg_train_file(project_dir, selected_subject)
        else:
            eeg_data_file = layout.get_eeg_test_file(project_dir, selected_subject)

        if not eeg_data_file:
            st.warning(f"No EEG data found in '{SUBJECT_DATA_DIR_PROCESSED}'")
            st.stop()
    st.session_state["selected_eeg_data_path"] = eeg_data_file
    st.session_state["selected_session"] = selected_session
    st.session_state["selected_repetition"] = repetition


def image_picker() -> None:
    image_categories = list(Path("data/images_THINGS").iterdir())
    selected_category = st.selectbox("Select a category", image_categories)

    if selected_category:
        IMAGE_FOLDER = Path("data/images_THINGS") / selected_category

        # --- List all image files ---
        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        image_files = [
            f for f in IMAGE_FOLDER.iterdir() if f.suffix.lower() in valid_ext
        ]

        if not image_files:
            st.warning(f"No images found in '{IMAGE_FOLDER}'")
            st.stop()

        # --- Display previews in a grid ---
        cols = st.columns(4)  # 4 thumbnails per row

        for idx, img_name in enumerate(image_files):
            with cols[idx % 4]:
                img_path = IMAGE_FOLDER / img_name
                img = Image.open(img_path)
                img.thumbnail((110, 110))
                st.image(img)
                if st.button("Select", key=img_name):
                    st.session_state.selected_image = img_name
                    st.session_state["selected_image_path"] = img_path


def raw_eeg_picker() -> None:
    subjects = list(Path("data/eeg_THINGS").iterdir())
    selected_subject = st.selectbox("Select a subject", sorted(subjects))
    selected_session = st.selectbox(
        "Select a session", ["ses-01", "ses-02", "ses-03", "ses-04"]
    )

    if selected_subject:
        EEG_FOLDER = Path("data/raw_eeg_THINGS") / selected_subject / selected_session

        eeg_data_file = [
            f
            for f in EEG_FOLDER.iterdir()
            if "training" in f.name.lower() and f.name.endswith(".npy")
        ]

        if not eeg_data_file:
            st.warning(f"No EEG data found in '{EEG_FOLDER}'")
            st.stop()

    st.session_state["selected_eeg_data_path"] = EEG_FOLDER / eeg_data_file[0]
