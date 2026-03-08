"""
THINGS-EEG2 Pipeline Dry Run Tests

Tests the --dry-run flag functionality for the complete pipeline.

Run with: pytest tests/test_preprocessing/test_pipeline_dryrun.py -v
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from things_eeg2_dataset.cli.main import EmbeddingModel
from things_eeg2_dataset.processing.pipeline import (
    PipelineConfig,
    ThingsEEGPipeline,
    _init_pipeline,
)

# Definition of a "fake" provisional environment for testing


@pytest.fixture  # Runs before the tests that require it
def mock_project_dir(tmp_path: Path) -> Path:
    """Create a mock project directory with expected structure."""
    (
        tmp_path / "raw_data"
    ).mkdir()  # Where tmp_path is a temporary path that is automatically cleaned up after the pipeline dry-run test
    (tmp_path / "processed").mkdir()
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "Image_set" / "training_images").mkdir(parents=True)
    (tmp_path / "Image_set" / "test_images").mkdir(parents=True)

    # Create mock raw data for test subjects
    for subject in [1, 2]:
        for session in range(1, 5):
            session_dir = (
                tmp_path / "raw_data" / f"sub-{subject:02d}" / f"ses-{session:02d}"
            )
            session_dir.mkdir(parents=True)
            (session_dir / "dummy.set").touch()

    return tmp_path


# Configuration Tests


class TestDryRunConfig:
    """Test dry_run flag configuration."""

    @pytest.mark.parametrize(
        "dry_run_value", [True, False]
    )  # Checks if the flag is correctly set for both True and False
    def test_dry_run_flag_initialization(
        self, mock_project_dir: Path, dry_run_value: bool
    ):
        """Test that the dry_run flag is correctly set."""
        config = PipelineConfig(
            project_dir=mock_project_dir,
            subjects=[1],
            dry_run=dry_run_value,
        )
        assert config.dry_run is dry_run_value

    def test_dry_run_default_is_false(self, mock_project_dir: Path):
        """Test dry_run defaults to False."""
        config = PipelineConfig(project_dir=mock_project_dir, subjects=[1])
        assert config.dry_run is False

    def test_init_pipeline_passes_dry_run(self, mock_project_dir: Path):
        """Test _init_pipeline correctly passes dry_run flag."""
        pipeline = _init_pipeline(
            project_dir=mock_project_dir,
            subjects=[1],
            overwrite=False,
            dry_run=True,
            skip_download=False,
            skip_preprocessing=False,
            skip_embeddings=False,
            interactive=False,
        )
        assert pipeline.cfg.dry_run is True


# Testing of how dry_run affects individual pipeline steps and overall behavior


class TestDryRunBehavior:
    """Test that dry_run changes pipeline behavior."""

    def test_prompts_auto_confirm_in_dry_run(self, mock_project_dir: Path):
        """Dry run mode auto-confirms all prompts."""
        config = PipelineConfig(
            project_dir=mock_project_dir,
            subjects=[1],
            dry_run=True,
        )
        pipeline = ThingsEEGPipeline(config)
        assert pipeline._prompt_user("test step") is True

    def test_prompts_require_input_normally(self, mock_project_dir: Path):
        """Normal mode requires user input for prompts."""
        config = PipelineConfig(
            project_dir=mock_project_dir,
            subjects=[1],
            dry_run=False,
        )
        pipeline = ThingsEEGPipeline(config)

        with patch("builtins.input", return_value="y"):
            assert pipeline._prompt_user("test") is True

        with patch("builtins.input", return_value="n"):
            assert pipeline._prompt_user("test") is False

    def test_version_file_not_written_in_dry_run(self, mock_project_dir: Path):
        """Version file is not written in dry run mode."""
        config = PipelineConfig(
            project_dir=mock_project_dir,
            subjects=[1],
            dry_run=True,
        )
        pipeline = ThingsEEGPipeline(config)
        version_file = config.project_dir / "data_version.txt"

        pipeline._write_version_file()

        # File should NOT exist in dry run mode
        assert not version_file.exists()

    def test_validation_warns_but_doesnt_fail(self, mock_project_dir: Path):
        config = PipelineConfig(
            project_dir=mock_project_dir,
            subjects=[1],
            dry_run=True,
        )
        pipeline = ThingsEEGPipeline(config)

        try:
            pipeline.validate_pipeline_outputs()
        except Exception as e:
            pytest.fail(f"Validation should not fail in dry run: {e}")


# Integration Tests


class TestDryRunIntegration:
    """Integration tests for full pipeline dry run."""

    @patch("things_eeg2_dataset.processing.pipeline.Downloader")
    @patch("things_eeg2_dataset.processing.pipeline.RawProcessor")
    @patch("things_eeg2_dataset.processing.pipeline.build_embedder")
    @patch("things_eeg2_dataset.processing.pipeline.validate_raw_data")
    @patch("things_eeg2_dataset.processing.pipeline.validate_processed_data")
    @patch("things_eeg2_dataset.processing.pipeline.torch")
    @patch("builtins.input", return_value="n")
    def test_full_pipeline_executes_all_stages(  # noqa: PLR0913
        self,
        mock_input,  # noqa: ANN001
        mock_torch,  # noqa: ANN001
        mock_validate_processed,  # noqa: ANN001
        mock_validate_raw,  # noqa: ANN001
        mock_build_embedder,  # noqa: ANN001
        mock_processor_class,  # noqa: ANN001
        mock_downloader_class,  # noqa: ANN001
        mock_project_dir: Path,
    ):
        """Full pipeline executes all stages (download, preprocessing and embedding generation) in dry run mode."""

        mock_downloader = Mock()
        mock_downloader.download_all.return_value = True
        mock_downloader_class.return_value = mock_downloader

        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_embedder = Mock()
        mock_build_embedder.return_value = mock_embedder

        mock_validation = Mock(has_failures=False, has_warnings=False, issues=[])
        mock_validate_raw.return_value = mock_validation
        mock_validate_processed.return_value = mock_validation

        mock_torch.cuda.is_available.return_value = False

        # Run pipeline
        config = PipelineConfig(
            project_dir=mock_project_dir,
            subjects=[1, 2],
            models=[
                EmbeddingModel.OPEN_CLIP_VIT_H_14
            ],  # Use of specific model for embedding generation (can be modified if required)
            dry_run=True,
            interactive=False,
        )

        pipeline = ThingsEEGPipeline(config)
        pipeline.run()

        # Verify all stages executed
        mock_downloader.download_all.assert_called_once()
        mock_processor.run.assert_called_once()
        mock_embedder.generate_and_store_embeddings.assert_called_once()

    def test_no_files_written_in_dry_run(self, mock_project_dir: Path):
        """Ensure that dry run produces no new files in processed directory."""
        processed_dir = mock_project_dir / "processed"

        # Count files before
        files_before = list(processed_dir.rglob("*"))
        files_before_count = len([f for f in files_before if f.is_file()])

        # Run in dry run mode using mocked components
        with (
            patch("things_eeg2_dataset.processing.pipeline.Downloader"),
            patch("things_eeg2_dataset.processing.pipeline.RawProcessor"),
            patch("things_eeg2_dataset.processing.pipeline.build_embedder"),
            patch(
                "things_eeg2_dataset.processing.pipeline.validate_raw_data"
            ) as mock_validate,
            patch("things_eeg2_dataset.processing.pipeline.validate_processed_data"),
            patch("things_eeg2_dataset.processing.pipeline.torch"),
            patch("builtins.input", return_value="n"),
        ):
            mock_validate.return_value = Mock(
                has_failures=False, has_warnings=False, issues=[]
            )

            config = PipelineConfig(
                project_dir=mock_project_dir,
                subjects=[1],
                dry_run=True,
                interactive=False,
            )

            pipeline = ThingsEEGPipeline(config)
            pipeline.run()

        # Count files after
        files_after = list(processed_dir.rglob("*"))
        files_after_count = len([f for f in files_after if f.is_file()])

        assert files_after_count == files_before_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
