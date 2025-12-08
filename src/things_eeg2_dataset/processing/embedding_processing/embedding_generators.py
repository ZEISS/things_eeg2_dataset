"""
Generate image and text embeddings for THINGS-EEG2 dataset.
"""

import argparse
from abc import ABC, abstractmethod
from collections.abc import Callable
from os import getenv
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    BatchFeature,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    Dinov2WithRegistersModel,
)

from .embedding_models import (
    ImageProjModel,
    IPAdapterPlusXL,
    IPAdapterXL,
    Resampler,
)

# Embedder names
OPEN_CLIP_VIT_H_14 = "open-clip-vit-h-14"
OPENAI_CLIP_VIT_L_14 = "openai-clip-vit-l-14"
DINOV2 = "dinov2"
IP_ADAPTER = "ip-adapter"


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation."""

    def __init__(
        self,
        data_path: Path | str = "data/things-eeg2/Image_set/",
        force: bool = False,
        dry_run: bool = False,
        device: str = "cuda:0",
    ) -> None:
        """
        Initializes the embedder.

        Args:
            data_path (Path | str): Path to the image dataset directory.
            force (bool): Whether to force reprocessing.
            dry_run (bool): Whether to simulate processing without executing.
            device (str): The device to run the model on.
        """
        self.data_path: Path = Path(data_path)
        self.train_images_path: Path = self.data_path / "training_images"
        self.test_images_path: Path = self.data_path / "test_images"
        self.embeds_dir: Path = self.data_path / "embeddings"
        self.embeds_dir.mkdir(parents=True, exist_ok=True)

        if not self.test_images_path.exists():
            raise FileNotFoundError(
                f"Image path {self.test_images_path} does not exist"
            )
        if not self.train_images_path.exists():
            raise FileNotFoundError(
                f"Image path {self.train_images_path} does not exist"
            )

        self.force: bool = force
        self.dry_run: bool = dry_run
        self.device: str = device
        self.model_type: str = "base"

    @staticmethod
    def get_texts(image_dir: Path | str) -> list[str]:
        """Extracts text descriptions from category directory names."""
        image_dir = Path(image_dir)
        text: list[str] = []
        category_dirs: list[Path] = sorted(image_dir.iterdir())
        for category_dir in category_dirs:
            cat_name: str = " ".join(category_dir.name.split("_")[1:])
            text.append(cat_name)
        return text

    @staticmethod
    def get_images(image_dir: Path | str) -> list[Path]:
        """Gathers all image paths from a directory."""
        image_dir = Path(image_dir)
        img_paths: list[Path] = []
        category_dirs: list[Path] = sorted(image_dir.iterdir())
        for category_dir in category_dirs:
            for image_file in sorted(category_dir.iterdir()):
                if image_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    im_path: Path = image_file
                    if not im_path.exists():
                        raise FileNotFoundError(f"Image path {im_path} does not exist")
                    img_paths.append(im_path)
        return img_paths

    def store_embeddings(
        self,
        image_dir: Path | str,
        embeds_path: Path | str,
        image2embed: Callable[[list[Image.Image]], Tensor],
        txt2embed: Callable[[list[str]], Tensor],
        embed_dim: tuple[int, ...],
    ) -> None:
        """
        Generates and stores image and text embeddings.

        Args:
            image_dir (Path | str): Path to the directory of images.
            embeds_path (Path | str): Path to save the embeddings.
            image2embed (Callable): Function to convert images to embeddings.
            txt2embed (Callable): Function to convert text to embeddings.
            embed_dim (tuple[int, ...]): The dimensions of the image embeddings.
        """
        image_dir = Path(image_dir)
        embeds_path = Path(embeds_path)
        Path.mkdir(Path(embeds_path).parent, parents=True, exist_ok=True)
        category_dirs: list[Path] = sorted(image_dir.iterdir())
        num_categories: int = len(category_dirs)
        num_images: int = len(list(category_dirs[0].iterdir()))
        img_embeds: Tensor = torch.zeros(
            num_categories * num_images, *embed_dim, dtype=torch.float32
        )
        text: list[str] = self.get_texts(image_dir)
        img_paths: list[Path] = self.get_images(image_dir)
        if len(img_paths) != num_categories * num_images:
            raise ValueError(
                f"Image paths do not match expected count: {len(img_paths)} but expected {num_categories * num_images}"
            )
        with torch.inference_mode():
            text_embeds: Tensor = txt2embed(text).cpu()
            batch_size = 40
            for i in tqdm(range(0, len(img_paths), batch_size)):
                batch_paths: list[Path] = img_paths[i : i + batch_size]
                images: list[Image.Image] = [
                    Image.open(path).convert("RGB") for path in batch_paths
                ]
                img_embeds[i : i + batch_size] = image2embed(images).cpu()
        print(f"Image embeddings shape: {img_embeds.shape}")
        print(f"Text embeddings shape: {text_embeds.shape}")

        if self.dry_run:
            print(f"Dry run enabled, not saving embeddings to {embeds_path}")
            return

        torch.save(
            {
                "img_features": img_embeds,
                "text_features": text_embeds,
            },
            embeds_path,
        )

    @abstractmethod
    def generate_and_store_embeddings(self) -> None:
        """An abstract method for generating and storing all embeddings."""
        raise NotImplementedError

    def _store_embeddings_if_needed(
        self,
        image_dir: Path | str,
        embeds_path: Path,
        image2embed: Callable[[list[Image.Image]], Tensor],
        txt2embed: Callable[[list[str]], Tensor],
        embed_dim: tuple[int, ...],
    ) -> None:
        if embeds_path.exists() and not self.force:
            print(f"Embeddings already exist at {embeds_path}, skipping.")
            return
        self.store_embeddings(image_dir, embeds_path, image2embed, txt2embed, embed_dim)


class OpenClipViTH14Embedder(BaseEmbedder):
    """A class to generate embeddings using OpenCLIP ViT-H-14 model."""

    def __init__(
        self,
        data_path: Path | str = "data/things-eeg2/Image_set/",
        force: bool = False,
        dry_run: bool = False,
        device: str = "cuda:0",
    ) -> None:
        """
        Initializes the embedder by loading the model and processors.

        Args:
            data_path (Path | str): Path to the image dataset directory.
            force (bool): Whether to force reprocessing.
            dry_run (bool): Whether to simulate processing without executing.
            device (str): The device to run the model on.
        """
        super().__init__(data_path, force, dry_run, device)
        self.model_type: str = "ViT-H-14"

        image_encoder: CLIPVisionModelWithProjection = (
            CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
            )
        )
        # We only need the pipeline to load the components easily
        pipeline: Any = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        ).to(self.device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
        )
        self.tokenizer: CLIPTokenizer = pipeline.tokenizer_2
        self.text_encoder: Any = pipeline.text_encoder_2
        self.feature_extractor: CLIPImageProcessor = pipeline.feature_extractor
        self.image_encoder: CLIPVisionModelWithProjection = pipeline.image_encoder

    def transform_clip_vis(self, images: list[Image.Image]) -> Tensor:
        """Generates pooled image embeddings."""
        images_processed: BatchFeature = self.feature_extractor(images)
        pixel_values: list[Any] = images_processed.pixel_values
        images_tensor: Tensor = torch.tensor(pixel_values, device=self.device)
        return self.image_encoder(images_tensor).image_embeds.half()

    def transform_clip_text(self, texts: list[str]) -> Tensor:
        """Generates pooled text embeddings."""
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_encoder(inputs.input_ids.to(self.device)).text_embeds.half()

    def transform_clip_vis_full(self, images: list[Image.Image]) -> Tensor:
        """Generates full sequence image embeddings."""
        images_processed = self.feature_extractor(images)
        pixel_values = images_processed.pixel_values
        images_tensor: Tensor = torch.tensor(pixel_values, device=self.device)
        return self.image_encoder(images_tensor).last_hidden_state.half()

    def transform_clip_text_full(self, texts: list[str]) -> Tensor:
        """Generates full sequence text embeddings."""
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_encoder(
            inputs.input_ids.to(self.device)
        ).last_hidden_state.half()

    def generate_and_store_embeddings(self) -> None:
        """Generates and stores all required embeddings for the model."""
        # Standard embedding
        train_embeds_path: Path = (
            self.embeds_dir / f"{self.model_type}_features_train.pt"
        )
        test_embeds_path: Path = self.embeds_dir / f"{self.model_type}_features_test.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.transform_clip_vis,
            self.transform_clip_text,
            embed_dim=(1024,),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.transform_clip_vis,
            self.transform_clip_text,
            embed_dim=(1024,),
        )

        # Full token embedding
        train_embeds_path = (
            self.embeds_dir / f"{self.model_type}_features_train_full.pt"
        )
        test_embeds_path = self.embeds_dir / f"{self.model_type}_features_test_full.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.transform_clip_vis_full,
            self.transform_clip_text_full,
            embed_dim=(
                257,
                1280,
            ),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.transform_clip_vis_full,
            self.transform_clip_text_full,
            embed_dim=(
                257,
                1280,
            ),
        )


class OpenAIClipVitL14Embedder(BaseEmbedder):
    """A class to generate embeddings using OpenAI's CLIP ViT-L-14 model."""

    def __init__(
        self,
        data_path: Path | str = "data/things-eeg2/Image_set/",
        force: bool = False,
        dry_run: bool = False,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(data_path, force, dry_run, device)
        self.model_type = "openai_ViT-L-14"
        self.processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.model: CLIPModel = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.model.to(device)

    def encode_text_pooled(self, text: list[str]) -> Tensor:
        """Generates pooled text embeddings."""
        batch_encoding: dict[str, Any] = self.tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens: Tensor = batch_encoding["input_ids"].to(self.device)
        return self.model.get_text_features(input_ids=tokens)

    def encode_vision_pooled(self, images: list[Image.Image]) -> Tensor:
        """Generates pooled image embeddings."""
        inputs = self.processor(images=images, return_tensors="pt")
        pixels: Tensor = inputs["pixel_values"].to(self.device)
        return self.model.get_image_features(pixel_values=pixels)

    def encode_text(self, text: list[str]) -> Tensor:
        """Generates full sequence text embeddings."""
        batch_encoding: dict[str, Any] = self.tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens: Tensor = batch_encoding["input_ids"].to(self.device)
        outputs: Any = self.model.text_model(input_ids=tokens)
        z: Tensor = self.model.text_projection(outputs.last_hidden_state)
        z_pooled: Tensor = self.model.text_projection(outputs.pooler_output)
        z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
        return z

    def encode_vision_noproj(self, images: list[Image.Image]) -> Tensor:
        """Helper for full sequence image embeddings without projection."""
        inputs = self.processor(images=images, return_tensors="pt")
        pixels: Tensor = inputs["pixel_values"].to(self.device)
        outputs: Any = self.model.vision_model(pixel_values=pixels)
        return outputs.last_hidden_state

    def encode_vision(self, images: list[Image.Image]) -> Tensor:
        """Generates full sequence image embeddings."""
        z: Tensor = self.encode_vision_noproj(images)
        z = self.model.vision_model.post_layernorm(z)
        z = self.model.visual_projection(z)
        z_pooled: Tensor = z[:, 0:1]
        z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z

    def generate_and_store_embeddings(self) -> None:
        """Generates and stores all required embeddings for the model."""
        # Standard embedding
        train_embeds_path: Path = (
            self.embeds_dir / f"{self.model_type}_features_train.pt"
        )
        test_embeds_path: Path = self.embeds_dir / f"{self.model_type}_features_test.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.encode_vision_pooled,
            self.encode_text_pooled,
            embed_dim=(768,),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.encode_vision_pooled,
            self.encode_text_pooled,
            embed_dim=(768,),
        )

        # Full token embedding
        train_embeds_path = (
            self.embeds_dir / f"{self.model_type}_features_train_full.pt"
        )
        test_embeds_path = self.embeds_dir / f"{self.model_type}_features_test_full.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.encode_vision,
            self.encode_text,
            embed_dim=(257, 768),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.encode_vision,
            self.encode_text,
            embed_dim=(257, 768),
        )


class DinoV2Embedder(BaseEmbedder):
    """A class to generate embeddings using DINOv2 with registers."""

    def __init__(
        self,
        data_path: Path | str = "data/things-eeg2/Image_set/",
        force: bool = False,
        dry_run: bool = False,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(data_path, force, dry_run, device)
        self.model_type = "dinov2-reg"
        self.image_processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-with-registers-base", use_fast=True
        )
        self.dinov2_model: Dinov2WithRegistersModel = (
            Dinov2WithRegistersModel.from_pretrained(
                "facebook/dinov2-with-registers-base"
            )
        )
        self.dinov2_model.to(device)

    def transform_dinov2_vis_full(self, images: list[Image.Image]) -> Tensor:
        """Generates full sequence image embeddings."""
        images_processed = self.image_processor(images, return_tensors="pt")
        pixel_values: Tensor = images_processed.pixel_values
        images_tensor: Tensor = torch.tensor(pixel_values, device=self.device)
        tokens: Tensor = self.dinov2_model(images_tensor).last_hidden_state
        return tokens.half()

    def transform_dinov2_vis_cls(self, images: list[Image.Image]) -> Tensor:
        """Generates CLS token image embeddings."""
        tokens: Tensor = self.transform_dinov2_vis_full(images)
        return tokens[:, 0, :]

    def transform_dinov2_vis_reg(self, images: list[Image.Image]) -> Tensor:
        """Generates CLS + register tokens image embeddings."""
        tokens: Tensor = self.transform_dinov2_vis_full(images)
        return tokens[:, 0:5, :]

    def transform_dinov2_text(self, texts: list[str]) -> Tensor:
        """Returns a dummy tensor as DINOv2 is vision-only."""
        print("WARNING: DINOv2 cannot be used for text, returning dummy tensor")
        return torch.tensor(0.0, device=self.device).half()

    def generate_and_store_embeddings(self) -> None:
        """Generates and stores all required embeddings for the model."""
        # CLS token embedding
        train_embeds_path: Path = (
            self.embeds_dir / f"{self.model_type}_features_train.pt"
        )
        test_embeds_path: Path = self.embeds_dir / f"{self.model_type}_features_test.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.transform_dinov2_vis_cls,
            self.transform_dinov2_text,
            embed_dim=(768,),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.transform_dinov2_vis_cls,
            self.transform_dinov2_text,
            embed_dim=(768,),
        )

        # CLS + Register tokens embedding
        train_embeds_path = (
            self.embeds_dir / f"{self.model_type}_features_train_registers.pt"
        )
        test_embeds_path = (
            self.embeds_dir / f"{self.model_type}_features_test_registers.pt"
        )
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.transform_dinov2_vis_reg,
            self.transform_dinov2_text,
            embed_dim=(
                5,
                768,
            ),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.transform_dinov2_vis_reg,
            self.transform_dinov2_text,
            embed_dim=(
                5,
                768,
            ),
        )

        # All tokens embedding
        train_embeds_path = (
            self.embeds_dir / f"{self.model_type}_features_train_full.pt"
        )
        test_embeds_path = self.embeds_dir / f"{self.model_type}_features_test_full.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.transform_dinov2_vis_full,
            self.transform_dinov2_text,
            embed_dim=(
                261,
                768,
            ),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.transform_dinov2_vis_full,
            self.transform_dinov2_text,
            embed_dim=(
                261,
                768,
            ),
        )


class IPAdapterEmbedder(BaseEmbedder):
    """A class to generate embeddings using IP-Adapter with ViT-H-14."""

    def __init__(
        self,
        data_path: Path | str = "data/things-eeg2/Image_set/",
        force: bool = False,
        dry_run: bool = False,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(data_path, force, dry_run, device)
        self.model_type = "ip-adapter-plus-vit-h-14"
        image_encoder: CLIPVisionModelWithProjection = (
            CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
            )
        )
        pipeline: Any = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        ).to(device)
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        )

        self.ip_adapter_proj: Resampler = self._load_ipa_proj_model(
            plus_model=True, device=device
        )
        self.ip_adapter_proj.to(device).eval()
        self.feature_extractor: CLIPImageProcessor = pipeline.feature_extractor
        self.image_encoder = pipeline.image_encoder

    def _load_ipa_proj_model(
        self, plus_model: bool = True, device: str = "cuda:0"
    ) -> ImageProjModel | Resampler:
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "models/image_encoder"

        if plus_model:
            num_tokens = 16
            fname = "ip-adapter-plus_sdxl_vit-h.safetensors"
        else:
            num_tokens = 4
            fname = "ip-adapter_sdxl_vit-h.safetensors"

        # load cached weights from huggingface hub
        hf_home = Path(getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
        weights_path = hf_home / "hub" / "models--h94--IP-Adapter" / "snapshots"
        weights_path = weights_path / "018e402774aeeddd60609b4ecdb7e298259dc729"
        weights_path = weights_path / "sdxl_models" / fname

        # Create the diffusion pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(
            base_model_path, torch_dtype=torch.float16
        )

        ip_model: IPAdapterPlusXL | IPAdapterXL

        if plus_model:
            ip_model = IPAdapterPlusXL(
                pipe,
                image_encoder_path,
                str(weights_path),
                device,
                num_tokens=num_tokens,
            )
        else:
            ip_model = IPAdapterXL(
                pipe,
                image_encoder_path,
                str(weights_path),
                device,
                num_tokens=num_tokens,
            )

        # extract image projection model from ip-adapter
        image_proj_model = ip_model.image_proj_model
        image_proj_model.requires_grad_(False)
        image_proj_model.float()  # convert weights to fp32

        del pipe
        torch.cuda.empty_cache()

        return image_proj_model

    def transform_ipa_vis(self, images: list[Image.Image]) -> Tensor:
        """Generates IP-Adapter image embeddings."""
        images_processed: BatchFeature = self.feature_extractor(
            images, return_tensors="pt"
        )
        pixel_values: Tensor = images_processed.pixel_values
        images_tensor: Tensor = torch.tensor(
            pixel_values, device=self.device, dtype=torch.float16
        )
        clip_embeds: Tensor = self.image_encoder(images_tensor).last_hidden_state
        ipa_embeds: Tensor = self.ip_adapter_proj(clip_embeds)
        return ipa_embeds.half()

    def transform_dummy_text(self, texts: list[str]) -> Tensor:
        """Returns a dummy tensor as IP-Adapter is vision-only."""
        print("WARNING: IP-Adapter is for images, returning dummy tensor for text")
        return torch.tensor(0.0, device=self.device).half()

    def generate_and_store_embeddings(self) -> None:
        """Generates and stores all required embeddings for the model."""
        train_embeds_path: Path = (
            self.embeds_dir / f"{self.model_type}_features_train.pt"
        )
        test_embeds_path: Path = self.embeds_dir / f"{self.model_type}_features_test.pt"
        self._store_embeddings_if_needed(
            self.test_images_path,
            test_embeds_path,
            self.transform_ipa_vis,
            self.transform_dummy_text,
            embed_dim=(16, 2048),
        )
        self._store_embeddings_if_needed(
            self.train_images_path,
            train_embeds_path,
            self.transform_ipa_vis,
            self.transform_dummy_text,
            embed_dim=(16, 2048),
        )


EMBEDDER_DICT = {
    OPEN_CLIP_VIT_H_14: OpenClipViTH14Embedder,
    OPENAI_CLIP_VIT_L_14: OpenAIClipVitL14Embedder,
    DINOV2: DinoV2Embedder,
    IP_ADAPTER: IPAdapterEmbedder,
}


def build_embedder(
    model_type: str,
    data_path: Path | str = "data/things-eeg2/Image_set/",
    force: bool = False,
    dry_run: bool = False,
    device: str = "cuda:0",
) -> BaseEmbedder:
    """Factory function to build the appropriate embedder based on model type."""
    if model_type not in EMBEDDER_DICT:
        raise ValueError(f"Unknown model type: {model_type}")
    embedder_class = EMBEDDER_DICT[model_type]
    return embedder_class(data_path, force, dry_run, device)  # type: ignore[abstract]


__all__ = [
    "BaseEmbedder",
    "DinoV2Embedder",
    "IPAdapterEmbedder",
    "OpenAIClipVitL14Embedder",
    "OpenClipViTH14Embedder",
    "build_embedder",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image and text embeddings for THINGS-EEG2 dataset."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[
            "open-clip-vit-h-14",
            "openai-clip-vit-l-14",
            "dinov2",
            "ip-adapter",
        ],
        required=True,
        help="Model type to use for generating embeddings.",
    )
    parser.add_argument(
        "-p",
        "--data_path",
        type=str,
        default="data/things-eeg2/Image_set/",
        help="Path to the image dataset directory.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force reprocessing of existing embeddings.",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="Simulate processing without saving embeddings.",
    )
    args: argparse.Namespace = parser.parse_args()

    embedder: BaseEmbedder = build_embedder(
        args.model,
        data_path=args.data_path,
        force=args.force,
        dry_run=args.dry_run,
        device="cuda:0",
    )
    embedder.generate_and_store_embeddings()
