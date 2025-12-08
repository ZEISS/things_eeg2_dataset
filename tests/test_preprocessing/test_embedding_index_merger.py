import filecmp
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

from things_eeg2_dataset.processing import EmbeddingIndexMerger


@pytest.fixture
def minimal_test_data(tmp_path: Path):
    """Create minimal test dataset with 2 subjects, 2 images."""
    # Create directory structure
    processed_dir = tmp_path / "processed"
    embeddings_dir = tmp_path / "embeddings"
    processed_dir.mkdir()
    embeddings_dir.mkdir()

    # Create training EEG index CSV (10 rows: 2 subjects x 2 images x 2 + 1 repetitions)
    train_data = {
        "global_index": list(range(10)),
        "subject": ["sub-01"] * 5 + ["sub-02"] * 5,
        "subject_pos": [0] * 5 + [1] * 5,
        "class_id": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        "sample_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "repetition": [0, 1, 2, 0, 1, 0, 1, 2, 0, 1],
        "image_index": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(processed_dir / "training_eeg_index.csv", index=False)

    # Create test EEG index CSV (4 rows: 2 subjects x 2 images, no repetitions in test)
    test_data = {
        "global_index": list(range(4)),
        "subject": ["sub-01", "sub-01", "sub-02", "sub-02"],
        "subject_pos": [0, 0, 1, 1],
        "class_id": [0, 1, 0, 1],
        "sample_id": [0, 0, 0, 0],
        "repetition": [-1, -1, -1, -1],  # Test uses averaged repetitions
        "image_index": [0, 1, 0, 1],
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(processed_dir / "test_eeg_index.csv", index=False)

    # Create embedding files for ViT-H-14
    # Training embeddings: 2 images
    train_embed_pooled = {
        "img_features": torch.randn(2, 1024),
        "text_features": torch.randn(2, 1024),
    }
    torch.save(train_embed_pooled, embeddings_dir / "ViT-H-14_features_train.pt")

    train_embed_full = {
        "img_features": torch.randn(2, 257, 1280),
        "text_features": torch.randn(2, 77, 1280),
    }
    torch.save(train_embed_full, embeddings_dir / "ViT-H-14_features_train_full.pt")

    # Test embeddings: 2 images
    test_embed_pooled = {
        "img_features": torch.randn(2, 1024),
        "text_features": torch.randn(2, 1024),
    }
    torch.save(test_embed_pooled, embeddings_dir / "ViT-H-14_features_test.pt")

    test_embed_full = {
        "img_features": torch.randn(2, 257, 1280),
        "text_features": torch.randn(2, 77, 1280),
    }
    torch.save(test_embed_full, embeddings_dir / "ViT-H-14_features_test_full.pt")

    return tmp_path


@pytest.fixture
def multimodel_test_data(tmp_path: Path):
    """Create test dataset with multiple embedding models."""
    processed_dir = tmp_path / "processed"
    embeddings_dir = tmp_path / "embeddings"
    processed_dir.mkdir()
    embeddings_dir.mkdir()

    # Create minimal EEG index
    train_data = {
        "global_index": list(range(4)),
        "subject": ["sub-01"] * 4,
        "subject_pos": [0] * 4,
        "class_id": [0, 0, 1, 1],
        "sample_id": [0, 0, 0, 0],
        "repetition": [0, 1, 0, 1],
        "image_index": [0, 0, 1, 1],
    }
    pd.DataFrame(train_data).to_csv(
        processed_dir / "training_eeg_index.csv", index=False
    )

    # Create embeddings for multiple models
    for model in ["ViT-H-14", "openai_ViT-L-14", "dinov2-reg"]:
        embed_data = {
            "img_features": torch.randn(2, 768),
            "text_features": torch.randn(2, 768),
        }
        torch.save(embed_data, embeddings_dir / f"{model}_features_train.pt")

    return tmp_path


# ============================================================================
# TC-1: Initialization Tests
# ============================================================================


def test_init_valid_paths(minimal_test_data: Path):
    """TC-1.1: Valid initialization with existing paths."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    assert merger.processed_dir.exists()
    assert merger.embeddings_dir.exists()
    assert merger.model_names == ["ViT-H-14"]


def test_init_nonexistent_processed_dir(minimal_test_data: Path):
    """TC-1.2: Reject non-existent processed directory."""
    with pytest.raises(FileNotFoundError):
        EmbeddingIndexMerger(
            processed_dir="/nonexistent/path",
            embeddings_dir=minimal_test_data / "embeddings",
            model_names=["ViT-H-14"],
        )


def test_init_nonexistent_embeddings_dir(minimal_test_data: Path):
    """TC-1.2: Reject non-existent embeddings directory."""
    with pytest.raises(FileNotFoundError):
        EmbeddingIndexMerger(
            processed_dir=minimal_test_data / "processed",
            embeddings_dir="/nonexistent/path",
            model_names=["ViT-H-14"],
        )


def test_init_empty_models(minimal_test_data: Path):
    """TC-1.3: Reject empty model names list."""
    with pytest.raises(ValueError, match=r"model_names.*empty"):
        EmbeddingIndexMerger(
            processed_dir=minimal_test_data / "processed",
            embeddings_dir=minimal_test_data / "embeddings",
            model_names=[],
        )


# ============================================================================
# TC-2: Embedding Detection Tests
# ============================================================================


def test_detect_pooled_embeddings(minimal_test_data: Path):
    """TC-2.1: Detect standard pooled embedding files."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    files = merger._detect_embedding_files("ViT-H-14", "training")
    assert files["pooled"] is not None
    assert files["pooled"].exists()
    assert "ViT-H-14_features_train.pt" in str(files["pooled"])


def test_detect_full_embeddings(minimal_test_data: Path):
    """TC-2.2: Detect full sequence embedding files."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    files = merger._detect_embedding_files("ViT-H-14", "training")
    assert files["full"] is not None
    assert files["full"].exists()
    assert "ViT-H-14_features_train_full.pt" in str(files["full"])


def test_detect_missing_embeddings(minimal_test_data: Path):
    """TC-2.3: Return None for missing embedding variants."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["NonexistentModel"],
    )
    files = merger._detect_embedding_files("NonexistentModel", "training")
    assert files["pooled"] is None
    assert files["full"] is None
    assert files["registers"] is None


def test_detect_test_partition_embeddings(minimal_test_data: Path):
    """TC-2: Detect embeddings for test partition."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    files = merger._detect_embedding_files("ViT-H-14", "test")
    assert files["pooled"] is not None
    assert "test" in str(files["pooled"])


# ============================================================================
# TC-3: CSV Backup Tests
# ============================================================================


def test_create_backup(minimal_test_data: Path):
    """TC-3.1: Create backup file before modification."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    original_csv = minimal_test_data / "processed" / "training_eeg_index.csv"
    merger._backup_csv(original_csv)

    backup_csv = Path(str(original_csv) + ".bak")
    assert backup_csv.exists()
    assert filecmp.cmp(original_csv, backup_csv)


def test_overwrite_backup(minimal_test_data: Path):
    """TC-3.2: Overwrite existing backup file."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    csv_path = minimal_test_data / "processed" / "training_eeg_index.csv"

    # Create first backup
    merger._backup_csv(csv_path)
    time.sleep(0.01)

    # Modify original
    df = pd.DataFrame({"new_col": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    # Create second backup
    merger._backup_csv(csv_path)

    # Backup should match current content
    backup = pd.read_csv(str(csv_path) + ".bak")
    assert "new_col" in backup.columns


# ============================================================================
# TC-4: Merge Embeddings Tests
# ============================================================================


def test_merge_single_model_training(minimal_test_data: Path):
    """TC-4.1: Merge single model for training partition."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("training")

    df = pd.read_csv(minimal_test_data / "processed" / "training_eeg_index.csv")

    # Check new columns exist
    assert "ViT-H-14_embed_path" in df.columns
    assert "ViT-H-14_embed_index" in df.columns
    assert "ViT-H-14_embed_available" in df.columns

    # Check full variant columns
    assert "ViT-H-14_full_embed_path" in df.columns
    assert "ViT-H-14_full_embed_index" in df.columns
    assert "ViT-H-14_full_embed_available" in df.columns

    # Check values are correct
    assert df["ViT-H-14_embed_available"].all()
    assert df["ViT-H-14_embed_index"].equals(df["image_index"])


def test_merge_multiple_models(multimodel_test_data: Path):
    """TC-4.2: Merge multiple models simultaneously."""
    merger = EmbeddingIndexMerger(
        processed_dir=multimodel_test_data / "processed",
        embeddings_dir=multimodel_test_data / "embeddings",
        model_names=["ViT-H-14", "openai_ViT-L-14"],
    )
    merger.merge_embeddings("training")

    df = pd.read_csv(multimodel_test_data / "processed" / "training_eeg_index.csv")

    # Check columns for both models
    assert "ViT-H-14_embed_path" in df.columns
    assert "openai_ViT-L-14_embed_path" in df.columns


def test_merge_test_partition(minimal_test_data: Path):
    """TC-4.3: Merge for test partition."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("test")

    df = pd.read_csv(minimal_test_data / "processed" / "test_eeg_index.csv")
    assert "ViT-H-14_embed_path" in df.columns
    assert "ViT-H-14_embed_available" in df.columns


def test_merge_missing_embeddings(minimal_test_data: Path):
    """TC-4.4: Handle missing embedding files gracefully."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["NonexistentModel"],
    )
    merger.merge_embeddings("training")

    df = pd.read_csv(minimal_test_data / "processed" / "training_eeg_index.csv")
    assert "NonexistentModel_embed_available" in df.columns
    assert not df["NonexistentModel_embed_available"].any()


def test_preserve_original_data(minimal_test_data: Path):
    """TC-4.5: Preserve original columns and values."""
    original_df = pd.read_csv(
        minimal_test_data / "processed" / "training_eeg_index.csv"
    )
    original_cols = set(original_df.columns)

    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("training")

    new_df = pd.read_csv(minimal_test_data / "processed" / "training_eeg_index.csv")

    # All original columns present
    assert original_cols.issubset(set(new_df.columns))

    # Original values unchanged
    for col in original_cols:
        pd.testing.assert_series_equal(original_df[col], new_df[col], check_names=True)


# ============================================================================
# TC-5: Validation Tests
# ============================================================================


def test_verify_alignment_valid(minimal_test_data: Path):
    """TC-5.1: Verify correct alignment passes validation."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("training")
    result = merger.verify_alignment("training")

    assert result["valid"] is True
    assert len(result["errors"]) == 0
    assert result["partition"] == "training"
    assert "ViT-H-14" in result["models_checked"]


def test_verify_alignment_out_of_bounds(minimal_test_data: Path):
    """TC-5.2: Detect image_index exceeding embedding tensor size."""
    # Modify CSV to have invalid image_index
    csv_path = minimal_test_data / "processed" / "training_eeg_index.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "image_index"] = 9999  # Out of bounds
    df.to_csv(csv_path, index=False)

    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("training")
    result = merger.verify_alignment("training")

    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert any(
        "out of bounds" in err.lower() or "index" in err.lower()
        for err in result["errors"]
    )


def test_verify_alignment_corrupted_file(minimal_test_data: Path):
    """TC-5.3: Handle corrupted embedding files gracefully."""
    # Create corrupted embedding file
    embed_path = minimal_test_data / "embeddings" / "ViT-H-14_features_train.pt"
    with embed_path.open("wb") as f:
        f.write(b"corrupted data")

    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("training")
    result = merger.verify_alignment("training")

    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_verify_unique_image_count(minimal_test_data: Path):
    """TC-5.4: Verify expected number of unique images."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )
    merger.merge_embeddings("training")
    result = merger.verify_alignment("training")

    # Minimal dataset has 2 unique images
    assert result["unique_images"] == 2


# ============================================================================
# TC-6: Edge Cases
# ============================================================================


def test_idempotent_merge(minimal_test_data: Path):
    """TC-6.1: Running merge twice produces same result."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )

    merger.merge_embeddings("training")
    df1 = pd.read_csv(minimal_test_data / "processed" / "training_eeg_index.csv")

    merger.merge_embeddings("training")
    df2 = pd.read_csv(minimal_test_data / "processed" / "training_eeg_index.csv")

    pd.testing.assert_frame_equal(df1, df2)


def test_invalid_partition_name(minimal_test_data: Path):
    """TC-6.2: Reject invalid partition names."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )

    with pytest.raises(ValueError, match="partition"):
        merger.merge_embeddings("invalid_partition")


def test_missing_eeg_index_csv(minimal_test_data: Path):
    """TC-6: Handle missing EEG index CSV."""
    # Remove the CSV
    (minimal_test_data / "processed" / "training_eeg_index.csv").unlink()

    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )

    with pytest.raises(FileNotFoundError):
        merger.merge_embeddings("training")


# ============================================================================
# TC-7: Integration Tests
# ============================================================================


def test_full_workflow_training(minimal_test_data: Path):
    """TC-7.1: Complete workflow for training partition."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )

    # Step 1: Merge
    merger.merge_embeddings("training")

    # Step 2: Verify
    result = merger.verify_alignment("training")

    # Assertions
    assert result["valid"] is True
    assert len(result["errors"]) == 0

    # Check CSV structure
    df = pd.read_csv(minimal_test_data / "processed" / "training_eeg_index.csv")
    expected_cols = [
        "ViT-H-14_embed_path",
        "ViT-H-14_embed_index",
        "ViT-H-14_embed_available",
        "ViT-H-14_full_embed_path",
        "ViT-H-14_full_embed_index",
        "ViT-H-14_full_embed_available",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_full_workflow_test(minimal_test_data: Path):
    """TC-7.2: Complete workflow for test partition."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )

    merger.merge_embeddings("test")
    result = merger.verify_alignment("test")

    assert result["valid"] is True
    assert result["partition"] == "test"


def test_backup_created_before_merge(minimal_test_data: Path):
    """TC: Verify backup is created before merging."""
    merger = EmbeddingIndexMerger(
        processed_dir=minimal_test_data / "processed",
        embeddings_dir=minimal_test_data / "embeddings",
        model_names=["ViT-H-14"],
    )

    csv_path = minimal_test_data / "processed" / "training_eeg_index.csv"
    backup_path = Path(str(csv_path) + ".bak")

    # Ensure no backup exists
    if backup_path.exists():
        backup_path.unlink()

    merger.merge_embeddings("training")

    # Backup should be created
    assert backup_path.exists()
