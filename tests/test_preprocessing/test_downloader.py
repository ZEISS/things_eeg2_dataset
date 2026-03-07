import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from things_eeg2_dataset.processing.downloading.downloader import Downloader


def create_mock_subject(root_dir: Path, subject_id: int, valid: bool = True) -> Path:
    """Helper to create a dummy subject directory structure on disk."""
    subject_str = f"sub-{subject_id:02d}"
    subject_path = root_dir / subject_str
    subject_path.mkdir(parents=True, exist_ok=True)

    num_sessions = 5 if valid else 1  # Create fewer sessions for invalid

    for session in range(1, num_sessions):
        session_dir = subject_path / f"ses-{session:02d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "raw_eeg_training.npy").write_bytes(b"dummy data")
        if valid:
            (session_dir / "raw_eeg_test.npy").write_bytes(b"dummy data")

    return subject_path


@pytest.fixture
def tmp_data_path(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    root = tmp_path / "things-eeg2"
    (root / "raw_data").mkdir(parents=True)
    return root


@pytest.fixture
def downloader(tmp_data_path: Path):
    """Create a Downloader instance for testing."""
    return Downloader(
        project_dir=tmp_data_path,
        subjects=[1, 2],
        timeout=10,
        max_retries=2,
    )


@pytest.fixture
def test_zip_file(tmp_path: Path) -> Path:
    """Create a small valid subject ZIP file."""
    zip_path = tmp_path / "sub-01.zip"

    # Build temp structure to zip
    build_dir = tmp_path / "build_zip"
    create_mock_subject(build_dir, 1, valid=True)

    # Create ZIP
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in build_dir.rglob("*"):
            if file.is_file():
                # Create arcname relative to build_dir
                zf.write(file, arcname=file.relative_to(build_dir))

    return zip_path


def test_init_default_parameters(tmp_path: Path) -> None:
    """Test initialization with default parameters."""
    d = Downloader(project_dir=tmp_path)

    assert d.project_dir == tmp_path
    assert d.raw_dir == tmp_path / "raw_data"
    # Ensure defaults are set
    assert d.subjects == list(range(1, 11))
    assert d.overwrite is False


def test_init_creates_directory_structure(tmp_path: Path) -> None:
    """Test that initialization creates required subdirectories."""
    data_path = tmp_path / "fresh_project"
    Downloader(project_dir=data_path)

    assert (data_path / "raw_data").exists()
    assert (data_path / "source_data").exists()
    assert (data_path / "Image_set").exists()


def test_init_dry_run_is_safe(tmp_path: Path) -> None:
    """Test that dry_run mode does not create directories."""
    data_path = tmp_path / "dry_run_project"
    Downloader(project_dir=data_path, dry_run=True)

    assert not (data_path / "raw_data").exists()


def test_check_if_exists_valid(downloader: Downloader) -> None:
    """Test checking existence with valid extracted structure."""
    create_mock_subject(downloader.raw_dir, 1, valid=True)

    zip_exists, extracted, valid = downloader._check_if_exists(1, downloader.raw_dir)
    assert (zip_exists, extracted, valid) == (False, True, True)


def test_check_if_exists_invalid(downloader: Downloader) -> None:
    """Test checking existence with incomplete structure."""
    create_mock_subject(downloader.raw_dir, 1, valid=False)

    _, extracted, valid = downloader._check_if_exists(1, downloader.raw_dir)
    assert extracted is True
    assert valid is False  # Should fail validation


def test_validate_structure_missing_files(downloader: Downloader) -> None:
    """Test validation logic specifically."""
    sub_path = create_mock_subject(downloader.raw_dir, 1, valid=True)

    # Delete a required file
    (sub_path / "ses-01" / "raw_eeg_training.npy").unlink()

    assert downloader._validate_subject_structure(sub_path) is False


def test_download_raw_data_summary(downloader: Downloader) -> None:
    """Test the high-level batch download function."""
    # Mock individual subject download to succeed
    with patch.object(downloader, "download_subject", return_value=True) as mock_dl:
        results = downloader.download_raw_data()

    assert results == {1: True, 2: True}
    assert mock_dl.call_count == 2
