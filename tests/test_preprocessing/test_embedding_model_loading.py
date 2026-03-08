from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import transformers
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from things_eeg2_dataset.cli.main import EmbeddingModel
from things_eeg2_dataset.processing.embedding_processing import (
    embedding_generators as embed_generators,
)
from things_eeg2_dataset.verification.dependency_ver import CRITICAL_DEPENDENCIES


class _DummyModule:
    def to(self, *_args: object, **_kwargs: object) -> "_DummyModule":
        return self

    def eval(self) -> "_DummyModule":
        return self

    def requires_grad_(self, *_args: object, **_kwargs: object) -> "_DummyModule":
        return self

    def half(self) -> "_DummyModule":
        return self


class _DummyTokenizer:
    model_max_length = 77


class _DummyPipeline(_DummyModule):
    def __init__(self) -> None:
        self.feature_extractor = object()
        self.image_encoder = _DummyModule()

    def load_ip_adapter(self, *_args: object, **_kwargs: object) -> None:
        return None


def _create_project_dir(tmp_path: Path) -> Path:
    (tmp_path / "Image_set" / "training_images").mkdir(parents=True)
    (tmp_path / "Image_set" / "test_images").mkdir(parents=True)
    return tmp_path


def _patch_embedder_dependencies(stack: ExitStack) -> None:
    stack.enter_context(
        patch.object(embed_generators.torch.cuda, "is_available", return_value=False)
    )
    stack.enter_context(
        patch.object(
            embed_generators.CLIPVisionModelWithProjection,
            "from_pretrained",
            return_value=_DummyModule(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.CLIPTokenizer,
            "from_pretrained",
            return_value=_DummyTokenizer(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.CLIPImageProcessor,
            "from_pretrained",
            return_value=object(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.CLIPModel,
            "from_pretrained",
            return_value=_DummyModule(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.AutoImageProcessor,
            "from_pretrained",
            return_value=object(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.Dinov2WithRegistersModel,
            "from_pretrained",
            return_value=_DummyModule(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.SiglipModel,
            "from_pretrained",
            return_value=_DummyModule(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.AutoPipelineForText2Image,
            "from_pretrained",
            return_value=_DummyPipeline(),
        )
    )
    stack.enter_context(
        patch.object(
            embed_generators.IPAdapterEmbedder,
            "_load_ipa_proj_model",
            return_value=_DummyModule(),
        )
    )
    stack.enter_context(
        patch.object(
            transformers.CLIPTextModelWithProjection,
            "from_pretrained",
            return_value=_DummyModule(),
        )
    )
    stack.enter_context(
        patch.object(
            transformers.AutoProcessor,
            "from_pretrained",
            return_value=object(),
        )
    )
    stack.enter_context(
        patch.object(
            transformers.AutoModel,
            "from_pretrained",
            return_value=_DummyModule(),
        )
    )


@pytest.mark.parametrize("model_type", list(EmbeddingModel))
def test_build_embedder_loads_all_supported_models(
    tmp_path: Path, model_type: EmbeddingModel
) -> None:
    project_dir = _create_project_dir(tmp_path)
    with ExitStack() as stack:
        _patch_embedder_dependencies(stack)
        embedder = embed_generators.build_embedder(
            model_type=model_type,
            project_dir=project_dir,
            device="cpu",
        )

    expected_type = embed_generators.EMBEDDER_DICT[model_type]
    assert isinstance(embedder, expected_type)


def test_transformers_dependency_accepts_newer_major_versions() -> None:
    specifier = SpecifierSet(CRITICAL_DEPENDENCIES["transformers"])

    assert Version("4.56.2") in specifier
    assert Version("4.70.0") in specifier
    assert Version("5.0.0") in specifier
    assert Version("5.3.1") in specifier
    assert Version("4.51.0") not in specifier
    assert Version("6.0.0") not in specifier


def test_diffusers_dependency_requires_transformers5_compatible_release() -> None:
    specifier = SpecifierSet(CRITICAL_DEPENDENCIES["diffusers"])

    assert Version("0.37.0") in specifier
    assert Version("0.36.0") not in specifier
    assert Version("0.38.0") not in specifier
