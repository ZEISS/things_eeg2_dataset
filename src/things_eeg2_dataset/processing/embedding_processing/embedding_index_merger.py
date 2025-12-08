"""
Embedding Index Merger Tool.

This module provides functionality to augment EEG index CSV files with references
to image embedding files, creating a unified mapping between EEG epochs, images,
and their embeddings.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch

logger = logging.getLogger(__name__)


class EmbeddingIndexMerger:
    """
    Merges embedding file paths and indices into EEG index CSVs.

    This class augments EEG index CSV files with metadata about available image
    embeddings, including file paths, tensor indices, and availability flags.
    Supports multiple embedding models and validates alignment between EEG epochs
    and embeddings.

    Parameters
    ----------
    processed_dir : Path | str
        Directory containing EEG index CSVs (e.g., 'processed/')
    embeddings_dir : Path | str
        Directory containing embedding .pt files
    model_names : list[str]
        List of embedding model names to process (e.g., ["ViT-H-14"])

    Attributes
    ----------
    processed_dir : Path
        Path to directory with EEG index CSVs
    embeddings_dir : Path
        Path to directory with embedding files
    model_names : list[str]
        List of model names to process

    Examples
    --------
    >>> merger = EmbeddingIndexMerger(
    ...     processed_dir="/path/to/processed",
    ...     embeddings_dir="/path/to/embeddings",
    ...     model_names=["ViT-H-14", "openai_ViT-L-14"]
    ... )
    >>> merger.merge_embeddings(partition="training")
    >>> result = merger.verify_alignment(partition="training")
    >>> print(f"Valid: {result['valid']}")
    """

    def __init__(
        self,
        processed_dir: Path | str,
        embeddings_dir: Path | str,
        model_names: list[str],
    ) -> None:
        """
        Initialize the EmbeddingIndexMerger.

        Parameters
        ----------
        processed_dir : Path | str
            Directory containing EEG index CSVs
        embeddings_dir : Path | str
            Directory containing embedding .pt files
        model_names : list[str]
            List of embedding model names to add to index

        Raises
        ------
        FileNotFoundError
            If processed_dir or embeddings_dir does not exist
        ValueError
            If model_names is empty
        """
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.model_names = model_names

        # Validation
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed directory not found: {self.processed_dir}"
            )
        if not self.embeddings_dir.exists():
            raise FileNotFoundError(
                f"Embeddings directory not found: {self.embeddings_dir}"
            )
        if not model_names:
            raise ValueError("model_names cannot be empty")

        logger.info("Initialized EmbeddingIndexMerger")
        logger.info(f"  Processed dir: {self.processed_dir}")
        logger.info(f"  Embeddings dir: {self.embeddings_dir}")
        logger.info(f"  Models: {self.model_names}")

    def merge_embeddings(self, partition: str) -> None:
        """
        Add embedding paths and indices to EEG index.

        Augments the EEG index CSV for the specified partition with embedding
        metadata for all configured models. Creates a backup of the original
        CSV before modification.

        Parameters
        ----------
        partition : str
            Either "training" or "test"

        Raises
        ------
        ValueError
            If partition is not "training" or "test"
        FileNotFoundError
            If EEG index CSV does not exist
        PermissionError
            If unable to write to CSV file

        Side Effects
        ------------
        Modifies the EEG index CSV in-place, adding columns:
        - `{model_name}_embed_path` : Path to embedding .pt file
        - `{model_name}_embed_index` : Index within embedding tensor
        - `{model_name}_embed_available` : Boolean flag

        Notes
        -----
        For models with multiple variants (pooled, full, registers), separate
        columns are added for each variant.
        """
        logger.info(f"\n=== Merging embeddings for {partition} partition ===")

        # Get CSV path
        csv_path = self._get_csv_path(partition)
        if not csv_path.exists():
            raise FileNotFoundError(f"EEG index CSV not found: {csv_path}")

        # Backup original CSV
        self._backup_csv(csv_path)
        logger.info(f"Created backup: {csv_path}.bak")

        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path.name}")

        # Add columns for each model
        for model_name in self.model_names:
            logger.info(f"Processing model: {model_name}")
            df = self._add_model_columns(df, model_name, partition)

        # Save updated CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved updated CSV with {len(df.columns)} columns")
        logger.info(f"Merge complete for {partition} partition\n")

    def verify_alignment(self, partition: str) -> dict[str, Any]:
        """
        Verify that image indices in EEG index match embedding file ordering.

        Validates that:
        1. All image_index values are within embedding tensor bounds
        2. Embedding files load successfully
        3. Expected number of unique images is present
        4. No data corruption or misalignment

        Parameters
        ----------
        partition : str
            Either "training" or "test"

        Returns
        -------
        dict[str, Any]
            Validation results with keys:
            - valid : bool - Overall validation status
            - partition : str - The partition validated
            - models_checked : list[str] - Models that were validated
            - total_rows : int - Total rows in EEG index
            - unique_images : int - Number of unique images
            - errors : list[str] - List of error messages
            - warnings : list[str] - List of warning messages

        Raises
        ------
        FileNotFoundError
            If EEG index CSV does not exist
        ValueError
            If partition is invalid

        Examples
        --------
        >>> result = merger.verify_alignment("training")
        >>> if not result['valid']:
        ...     for error in result['errors']:
        ...         print(f"Error: {error}")
        """
        logger.info(f"\n=== Verifying alignment for {partition} partition ===")

        csv_path = self._get_csv_path(partition)
        if not csv_path.exists():
            raise FileNotFoundError(f"EEG index CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        errors = []
        warnings = []
        models_checked = []

        # Check unique images
        unique_images = df["image_index"].nunique()
        expected_count = self._get_expected_image_count(partition)

        if unique_images != expected_count:
            warnings.append(
                f"Expected {expected_count} unique images, found {unique_images}"
            )

        # Validate each model
        for model_name in self.model_names:
            models_checked.append(model_name)
            files = self._detect_embedding_files(model_name, partition)

            for variant, file_path in files.items():
                if file_path is None:
                    warnings.append(f"No {variant} embedding file for {model_name}")
                    continue

                # Try to load embedding file
                embed_data = self._load_embedding_file(file_path)
                if embed_data is None:
                    errors.append(
                        f"Failed to load {variant} embeddings for {model_name}: {file_path}"
                    )
                    continue

                # Validate indices
                is_valid, validation_errors = self._validate_image_indices(
                    df, embed_data, model_name, variant
                )
                if not is_valid:
                    errors.extend(validation_errors)

        is_valid = len(errors) == 0

        result = {
            "valid": is_valid,
            "partition": partition,
            "models_checked": models_checked,
            "total_rows": len(df),
            "unique_images": unique_images,
            "errors": errors,
            "warnings": warnings,
        }

        if is_valid:
            logger.info(f"✓ Validation PASSED for {partition} partition")
        else:
            logger.error(f"✗ Validation FAILED for {partition} partition")
            for error in errors:
                logger.error(f"  - {error}")

        if warnings:
            for warning in warnings:
                logger.warning(f"  - {warning}")

        return result

    def _detect_embedding_files(
        self, model_name: str, partition: str
    ) -> dict[str, Path | None]:
        """
        Detect available embedding files for a model and partition.

        Searches for embedding files with standard naming patterns:
        - Pooled: {model}_features_{train|test}.pt
        - Full: {model}_features_{train|test}_full.pt
        - Registers: {model}_features_{train|test}_registers.pt (DINOv2)

        Parameters
        ----------
        model_name : str
            Name of the embedding model
        partition : str
            Either "training" or "test"

        Returns
        -------
        dict[str, Path | None]
            Dictionary mapping variant names to file paths:
            - "pooled" : Path to pooled embeddings or None
            - "full" : Path to full sequence embeddings or None
            - "registers" : Path to register tokens or None

        Notes
        -----
        Returns None for variants that are not found. Does not raise errors
        for missing files.
        """
        # Map partition to file suffix
        suffix = "train" if partition == "training" else "test"

        # Search for different variants
        variants = {
            "pooled": self.embeddings_dir / f"{model_name}_features_{suffix}.pt",
            "full": self.embeddings_dir / f"{model_name}_features_{suffix}_full.pt",
            "registers": self.embeddings_dir
            / f"{model_name}_features_{suffix}_registers.pt",
        }

        # Check which files exist
        result = {}
        for variant_name, file_path in variants.items():
            result[variant_name] = file_path if file_path.exists() else None

        return result

    def _backup_csv(self, csv_path: Path) -> None:
        """
        Create a backup of the CSV file before modification.

        Creates a backup with '.bak' extension. If a backup already exists,
        it will be overwritten.

        Parameters
        ----------
        csv_path : Path
            Path to the CSV file to backup

        Raises
        ------
        FileNotFoundError
            If csv_path does not exist
        PermissionError
            If unable to write backup file

        Side Effects
        ------------
        Creates {csv_path}.bak file
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        backup_path = Path(str(csv_path) + ".bak")
        shutil.copy2(csv_path, backup_path)

    def _load_embedding_file(self, embed_path: Path) -> dict[str, torch.Tensor] | None:
        """
        Load an embedding .pt file safely.

        Attempts to load a PyTorch .pt file containing embeddings. Returns None
        if the file is corrupted or cannot be loaded.

        Parameters
        ----------
        embed_path : Path
            Path to the embedding .pt file

        Returns
        -------
        dict[str, torch.Tensor] | None
            Dictionary with keys:
            - "img_features" : Tensor of image embeddings
            - "text_features" : Tensor of text embeddings
            Returns None if file cannot be loaded

        Notes
        -----
        Logs warnings if file cannot be loaded but does not raise exceptions.
        """
        try:
            data = torch.load(embed_path, map_location="cpu")
            if not isinstance(data, dict):
                logger.warning(f"Embedding file has unexpected format: {embed_path}")
                return None
            if "img_features" not in data:
                logger.warning(f"Embedding file missing 'img_features': {embed_path}")
                return None
            return data
        except Exception as e:
            logger.warning(f"Failed to load embedding file {embed_path}: {e}")
            return None

    def _add_model_columns(
        self,
        df: pd.DataFrame,
        model_name: str,
        partition: str,
    ) -> pd.DataFrame:
        """
        Add embedding columns for a specific model to the dataframe.

        For each detected embedding variant (pooled, full, registers), adds
        three columns: path, index, and availability flag.

        Parameters
        ----------
        df : pd.DataFrame
            The EEG index dataframe to augment
        model_name : str
            Name of the embedding model
        partition : str
            Either "training" or "test"

        Returns
        -------
        pd.DataFrame
            Augmented dataframe with new embedding columns

        Notes
        -----
        Modifies the dataframe in-place but also returns it for chaining.
        Sets embed_available=False if embedding files are missing or invalid.
        """
        files = self._detect_embedding_files(model_name, partition)

        # Process each variant
        for variant, file_path in files.items():
            # Column name prefix
            if variant == "pooled":
                col_prefix = model_name
            else:
                col_prefix = f"{model_name}_{variant}"

            if file_path is None:
                # No embedding file found
                df[f"{col_prefix}_embed_path"] = ""
                df[f"{col_prefix}_embed_index"] = -1
                df[f"{col_prefix}_embed_available"] = False
                logger.info(f"  - {variant}: Not found")
            else:
                # Embedding file exists
                # Make path relative to embeddings_dir
                try:
                    rel_path = file_path.relative_to(self.embeddings_dir.parent)
                except ValueError:
                    rel_path = file_path

                df[f"{col_prefix}_embed_path"] = str(rel_path)
                df[f"{col_prefix}_embed_index"] = df["image_index"]
                df[f"{col_prefix}_embed_available"] = True
                logger.info(f"  - {variant}: {file_path.name}")

        return df

    def _validate_image_indices(
        self,
        df: pd.DataFrame,
        embed_data: dict[str, torch.Tensor],
        model_name: str,
        variant: str,
    ) -> tuple[bool, list[str]]:
        """
        Validate that image indices are within embedding tensor bounds.

        Parameters
        ----------
        df : pd.DataFrame
            EEG index dataframe
        embed_data : dict[str, torch.Tensor]
            Loaded embedding data
        model_name : str
            Name of the model
        variant : str
            Embedding variant (e.g., "pooled", "full")

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, error_messages)
            - is_valid: True if all indices are valid
            - error_messages: List of error descriptions if invalid

        Notes
        -----
        Checks that all image_index values are < tensor size along dimension 0.
        """
        errors = []

        img_features = embed_data["img_features"]
        num_embeddings = img_features.shape[0]

        # Check all indices are within bounds
        max_index = df["image_index"].max()
        min_index = df["image_index"].min()

        if max_index >= num_embeddings:
            errors.append(
                f"{model_name} ({variant}): image_index {max_index} out of bounds "
                f"(embedding tensor has {num_embeddings} images)"
            )

        if min_index < 0:
            errors.append(
                f"{model_name} ({variant}): negative image_index {min_index} found"
            )

        return len(errors) == 0, errors

    def _get_expected_image_count(self, partition: str) -> int:
        """
        Get the expected number of unique images for a partition.

        Parameters
        ----------
        partition : str
            Either "training" or "test"

        Returns
        -------
        int
            Expected number of unique images:
            - Training: 16540 (1654 classes x 10 samples)
            - Test: 200 (200 classes x 1 sample)
        """
        if partition == "training":
            return 1654 * 10  # 16540
        else:
            return 200

    def _get_csv_path(self, partition: str) -> Path:
        """
        Get the path to the EEG index CSV for a partition.

        Parameters
        ----------
        partition : str
            Either "training" or "test"

        Returns
        -------
        Path
            Path to the CSV file

        Raises
        ------
        ValueError
            If partition is not "training" or "test"
        """
        if partition not in ["training", "test"]:
            raise ValueError(
                f"Invalid partition: {partition}. Must be 'training' or 'test'"
            )

        filename = f"{partition}_eeg_index.csv"
        return self.processed_dir / filename
