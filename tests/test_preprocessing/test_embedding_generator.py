"""
Comprehensive test suite for embedding generators.

Tests cover initialization, static methods, embedding storage, and factory functions.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from things_eeg2_dataset.processing.embedding_processing import (
    BaseEmbedder,
    DinoV2Embedder,
    IPAdapterEmbedder,
    OpenAIClipVitL14Embedder,
    OpenClipViTH14Embedder,
    build_embedder,
)

# ============================================================================
# Concrete Implementation for Testing Abstract Base Class
# ============================================================================


class ConcreteEmbedder(BaseEmbedder):
    """Helper class to instantiate abstract BaseEmbedder."""

    def generate_and_store_embeddings(self) -> None:
        """Dummy implementation."""
        pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_dataset_structure(tmp_path: Path) -> Path:
    """Create a mock dataset directory structure."""
    data_path = tmp_path / "test_data"
    train_path = data_path / "training_images"
    test_path = data_path / "test_images"

    # Create category directories with images
    for split_path in [train_path, test_path]:
        for category in ["01_cat", "02_dog"]:
            cat_dir = split_path / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy image files
            for j in range(2):
                img_file = cat_dir / f"image_{j}.png"
                # Create a small dummy image
                img = Image.new("RGB", (10, 10), color="red")
                img.save(img_file)

    return data_path


@pytest.fixture
def mock_dependencies() -> Generator[None, None, None]:
    """
    Mock all heavy external dependencies (Transformers, Diffusers, Torch).

    This prevents importing heavy libraries and allows testing logic in isolation.
    """
    with (
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.CLIPVisionModelWithProjection"
        ) as mock_clip_vis,
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.AutoPipelineForText2Image"
        ) as mock_pipeline,
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.CLIPModel"
        ),
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.CLIPTokenizer"
        ),
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.CLIPImageProcessor"
        ),
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.Dinov2WithRegistersModel"
        ),
        patch(
            "things_eeg2_dataset.processing.embedding_processing.embedding_generators.AutoImageProcessor"
        ),
    ):
        # Setup common return values to avoid AttributeError
        mock_pipe_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_clip_vis.from_pretrained.return_value = MagicMock()

        yield


@pytest.fixture
def mock_ip_adapter_loader() -> Generator[MagicMock, None, None]:
    """Mock the internal IP Adapter loader method."""
    with patch.object(IPAdapterEmbedder, "_load_ipa_proj_model") as mock_load:
        mock_instance = MagicMock()
        mock_instance.to.return_value = mock_instance
        mock_load.return_value = mock_instance
        yield mock_load


# ============================================================================
# Test Class: TestBaseEmbedder
# ============================================================================


class TestBaseEmbedder:
    """Tests for BaseEmbedder initialization and core functionality."""

    def test_base_embedder_init_with_custom_path(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test that BaseEmbedder accepts and stores custom data_path."""
        embedder = ConcreteEmbedder(data_path=mock_dataset_structure)

        assert embedder.data_path == mock_dataset_structure
        assert embedder.train_images_path == mock_dataset_structure / "training_images"
        assert embedder.test_images_path == mock_dataset_structure / "test_images"
        assert embedder.embeds_dir == mock_dataset_structure / "embeddings"

    def test_base_embedder_creates_embeds_dir(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test that embeddings directory is created on init."""
        embeds_dir = mock_dataset_structure / "embeddings"
        # Ensure it doesn't exist before test
        if embeds_dir.exists():
            embeds_dir.rmdir()

        ConcreteEmbedder(data_path=mock_dataset_structure)

        assert embeds_dir.exists()

    def test_base_embedder_raises_on_missing_test_images(self, tmp_path: Path) -> None:
        """Test FileNotFoundError when test_images path doesn't exist."""
        data_path = tmp_path / "test_data"
        train_path = data_path / "training_images"
        train_path.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="test_images"):
            ConcreteEmbedder(data_path=data_path)

    def test_base_embedder_raises_on_missing_train_images(self, tmp_path: Path) -> None:
        """Test FileNotFoundError when training_images path doesn't exist."""
        data_path = tmp_path / "test_data"
        test_path = data_path / "test_images"
        test_path.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="training_images"):
            ConcreteEmbedder(data_path=data_path)

    def test_base_embedder_stores_parameters(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test that force, dry_run, device parameters are stored."""
        embedder = ConcreteEmbedder(
            data_path=mock_dataset_structure,
            force=True,
            dry_run=True,
            device="cuda:1",
        )

        assert embedder.force is True
        assert embedder.dry_run is True
        assert embedder.device == "cuda:1"


# ============================================================================
# Test Class: TestStaticMethods
# ============================================================================


class TestStaticMethods:
    """Tests for static helper methods."""

    def test_get_texts_extracts_category_names(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test get_texts extracts category names from directories."""
        train_path = mock_dataset_structure / "training_images"
        texts = BaseEmbedder.get_texts(train_path)

        assert len(texts) == 2
        assert texts[0] == "cat"
        assert texts[1] == "dog"

    def test_get_texts_handles_underscores(self, tmp_path: Path) -> None:
        """Test underscores in category names are converted to spaces."""
        image_dir = tmp_path / "images"
        (image_dir / "01_hello_world").mkdir(parents=True)
        (image_dir / "02_foo_bar_baz").mkdir(parents=True)

        texts = BaseEmbedder.get_texts(image_dir)

        assert texts[0] == "hello world"
        assert texts[1] == "foo bar baz"

    def test_get_images_collects_all_images(self, mock_dataset_structure: Path) -> None:
        """Test get_images finds all image files."""
        train_path = mock_dataset_structure / "training_images"
        images = BaseEmbedder.get_images(train_path)

        assert len(images) == 4  # 2 categories x 2 images
        assert all(isinstance(img, Path) for img in images)
        assert all(img.exists() for img in images)

    def test_get_images_filters_by_extension(self, tmp_path: Path) -> None:
        """Test only valid image extensions are included."""
        image_dir = tmp_path / "images"
        cat_dir = image_dir / "01_cat"
        cat_dir.mkdir(parents=True)

        # Create various file types
        (cat_dir / "image1.png").touch()
        (cat_dir / "image2.jpg").touch()
        (cat_dir / "image3.jpeg").touch()
        (cat_dir / "not_image.txt").touch()
        (cat_dir / "data.json").touch()

        images = BaseEmbedder.get_images(image_dir)

        assert len(images) == 3
        assert all(img.suffix.lower() in [".png", ".jpg", ".jpeg"] for img in images)

    def test_get_images_case_insensitive(self, tmp_path: Path) -> None:
        """Test image extension matching is case-insensitive."""
        image_dir = tmp_path / "images"
        cat_dir = image_dir / "01_cat"
        cat_dir.mkdir(parents=True)

        # Create images with uppercase extensions
        (cat_dir / "image1.PNG").touch()
        (cat_dir / "image2.JPG").touch()
        (cat_dir / "image3.JPEG").touch()

        images = BaseEmbedder.get_images(image_dir)

        assert len(images) == 3


class TestEmbeddingStorage:
    """Tests for embedding storage logic."""

    def test_store_embeddings_creates_parent_dirs(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test parent directories are created for embedding files."""
        embedder = ConcreteEmbedder(data_path=mock_dataset_structure)

        # Create a deeply nested path
        embeds_path = (
            mock_dataset_structure / "deep" / "nested" / "path" / "embeddings.pt"
        )

        def mock_img2embed(images: list[str]) -> torch.Tensor:
            return torch.randn(len(images), 768)

        def mock_txt2embed(texts: list[str]) -> torch.Tensor:
            return torch.randn(len(texts), 768)

        train_path = mock_dataset_structure / "training_images"

        embedder.store_embeddings(
            train_path,
            embeds_path,
            mock_img2embed,
            mock_txt2embed,
            embed_dim=(768,),
        )

        assert embeds_path.parent.exists()
        assert embeds_path.exists()

    def test_store_embeddings_saves_correct_format(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test saved file has correct dictionary structure."""
        embedder = ConcreteEmbedder(data_path=mock_dataset_structure)
        embeds_path = mock_dataset_structure / "embeddings" / "test.pt"

        def mock_img2embed(images: list[Path]) -> torch.Tensor:
            return torch.randn(len(images), 768)

        def mock_txt2embed(texts: list[str]) -> torch.Tensor:
            return torch.randn(len(texts), 768)

        train_path = mock_dataset_structure / "training_images"

        embedder.store_embeddings(
            train_path,
            embeds_path,
            mock_img2embed,
            mock_txt2embed,
            embed_dim=(768,),
        )

        # Load and verify
        saved_data = torch.load(embeds_path)
        assert "img_features" in saved_data
        assert "text_features" in saved_data
        assert saved_data["img_features"].shape == (4, 768)
        assert saved_data["text_features"].shape == (2, 768)

    def test_store_embeddings_dry_run_no_save(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test dry_run mode doesn't save to disk."""
        embedder = ConcreteEmbedder(data_path=mock_dataset_structure, dry_run=True)
        embeds_path = mock_dataset_structure / "embeddings" / "test.pt"

        def mock_img2embed(images: list[Path]) -> torch.Tensor:
            return torch.randn(len(images), 768)

        def mock_txt2embed(texts: list[str]) -> torch.Tensor:
            return torch.randn(len(texts), 768)

        train_path = mock_dataset_structure / "training_images"

        embedder.store_embeddings(
            train_path,
            embeds_path,
            mock_img2embed,
            mock_txt2embed,
            embed_dim=(768,),
        )

        assert not embeds_path.exists()

    def test_store_embeddings_if_needed_skips_existing(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test skipping when embeddings exist and force=False."""
        embedder = ConcreteEmbedder(data_path=mock_dataset_structure, force=False)

        embeds_path = mock_dataset_structure / "embeddings" / "existing.pt"
        embeds_path.touch()

        embedder.store_embeddings = Mock()  # type: ignore[assignment]

        train_path = mock_dataset_structure / "training_images"

        embedder._store_embeddings_if_needed(
            train_path, embeds_path, Mock(), Mock(), embed_dim=(768,)
        )

        embedder.store_embeddings.assert_not_called()

    def test_store_embeddings_if_needed_force_overwrites(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test force=True processes even when file exists."""
        embedder = ConcreteEmbedder(data_path=mock_dataset_structure, force=True)

        embeds_path = mock_dataset_structure / "embeddings" / "existing.pt"
        embeds_path.touch()

        embedder.store_embeddings = Mock()  # type: ignore[assignment]

        train_path = mock_dataset_structure / "training_images"

        embedder._store_embeddings_if_needed(
            train_path, embeds_path, Mock(), Mock(), embed_dim=(768,)
        )

        embedder.store_embeddings.assert_called_once()


# ============================================================================
# Test Class: TestEmbedderChildren
# ============================================================================


@pytest.mark.usefixtures("mock_dependencies")
class TestEmbedderChildren:
    """Tests for child embedder classes."""

    def test_open_clip_vith14_init_accepts_data_path(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test OpenClipViTH14Embedder accepts data_path."""
        embedder = OpenClipViTH14Embedder(data_path=mock_dataset_structure)

        assert embedder.data_path == mock_dataset_structure
        assert embedder.model_type == "ViT-H-14"

    def test_openai_clip_vitl14_init_accepts_data_path(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test OpenAIClipVitL14Embedder accepts data_path."""
        embedder = OpenAIClipVitL14Embedder(data_path=mock_dataset_structure)

        assert embedder.data_path == mock_dataset_structure
        assert embedder.model_type == "openai_ViT-L-14"

    def test_dinov2_init_accepts_data_path(self, mock_dataset_structure: Path) -> None:
        """Test DinoV2Embedder accepts data_path."""
        embedder = DinoV2Embedder(data_path=mock_dataset_structure)

        assert embedder.data_path == mock_dataset_structure
        assert embedder.model_type == "dinov2-reg"

    def test_ip_adapter_init_accepts_data_path(
        self, mock_ip_adapter_loader: MagicMock, mock_dataset_structure: Path
    ) -> None:
        """Test IPAdapterEmbedder accepts data_path."""
        embedder = IPAdapterEmbedder(data_path=mock_dataset_structure)

        assert embedder.data_path == mock_dataset_structure
        assert embedder.model_type == "ip-adapter-plus-vit-h-14"

    def test_each_child_has_unique_model_type(
        self, mock_ip_adapter_loader: MagicMock, mock_dataset_structure: Path
    ) -> None:
        """Test each embedder class has a unique model_type."""
        embedders = [
            OpenClipViTH14Embedder(data_path=mock_dataset_structure),
            OpenAIClipVitL14Embedder(data_path=mock_dataset_structure),
            DinoV2Embedder(data_path=mock_dataset_structure),
            IPAdapterEmbedder(data_path=mock_dataset_structure),
        ]

        model_types = [e.model_type for e in embedders]
        assert len(model_types) == len(set(model_types))


# ============================================================================
# Test Class: TestFactoryFunction
# ============================================================================


@pytest.mark.usefixtures("mock_dependencies")
class TestFactoryFunction:
    """Tests for the build_embedder factory function."""

    def test_build_embedder_creates_open_clip(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test factory creates OpenClipViTH14Embedder."""
        embedder = build_embedder(
            "open-clip-vit-h-14", data_path=mock_dataset_structure
        )
        assert isinstance(embedder, OpenClipViTH14Embedder)

    def test_build_embedder_creates_openai_clip(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test factory creates OpenAIClipVitL14Embedder."""
        embedder = build_embedder(
            "openai-clip-vit-l-14", data_path=mock_dataset_structure
        )
        assert isinstance(embedder, OpenAIClipVitL14Embedder)

    def test_build_embedder_creates_dinov2(self, mock_dataset_structure: Path) -> None:
        """Test factory creates DinoV2Embedder."""
        embedder = build_embedder("dinov2", data_path=mock_dataset_structure)
        assert isinstance(embedder, DinoV2Embedder)

    def test_build_embedder_creates_ip_adapter(
        self, mock_ip_adapter_loader: MagicMock, mock_dataset_structure: Path
    ) -> None:
        """Test factory creates IPAdapterEmbedder."""
        embedder = build_embedder("ip-adapter", data_path=mock_dataset_structure)
        assert isinstance(embedder, IPAdapterEmbedder)

    def test_build_embedder_passes_parameters(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test factory passes all parameters to constructor."""
        embedder = build_embedder(
            "openai-clip-vit-l-14",
            data_path=mock_dataset_structure,
            force=True,
            dry_run=True,
            device="cuda:1",
        )

        assert embedder.data_path == mock_dataset_structure
        assert embedder.force is True
        assert embedder.dry_run is True
        assert embedder.device == "cuda:1"

    def test_build_embedder_raises_on_unknown_type(
        self, mock_dataset_structure: Path
    ) -> None:
        """Test ValueError for unknown model_type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            build_embedder("invalid-model-type", data_path=mock_dataset_structure)


# ============================================================================
# Test Class: TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_store_embeddings_raises_on_image_count_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Test ValueError when image count doesn't match expected."""
        # Create mismatched structure
        data_path = tmp_path / "test_data"
        train_path = data_path / "training_images"

        cat1_dir = train_path / "01_cat"
        cat1_dir.mkdir(parents=True)
        (cat1_dir / "image1.png").touch()
        (cat1_dir / "image2.png").touch()

        cat2_dir = train_path / "02_dog"
        cat2_dir.mkdir(parents=True)
        (cat2_dir / "image1.png").touch()
        # Mismatch: Only one image in second category

        test_path = data_path / "test_images"
        test_path.mkdir(parents=True)
        (test_path / "01_cat").mkdir()
        (test_path / "02_dog").mkdir()

        embedder = ConcreteEmbedder(data_path=data_path)
        embeds_path = data_path / "embeddings" / "test.pt"

        def mock_img2embed(images: list[str]) -> torch.Tensor:
            return torch.randn(len(images), 768)

        def mock_txt2embed(texts: list[str]) -> torch.Tensor:
            return torch.randn(len(texts), 768)

        with pytest.raises(ValueError, match="Image paths do not match expected count"):
            embedder.store_embeddings(
                train_path,
                embeds_path,
                mock_img2embed,
                mock_txt2embed,
                embed_dim=(768,),
            )
