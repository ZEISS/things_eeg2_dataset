from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import loadfile
from things_eeg2_raw_processing.embedding_processing.embedding_models.ip_adapter.attention_processor import (
    AttnProcessor,
    IPAttnProcessor,
)
from things_eeg2_raw_processing.embedding_processing.embedding_models.ip_adapter.attention_processor_faceid import (
    LoRAIPAttnProcessor,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .resampler import FeedForward, PerceiverAttention
from .utils import get_generator, is_torch2_available

USE_DAFAULT_ATTN = False  # should be True for visualization_attnmap
if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from .attention_processor_faceid import (  # type: ignore[assignment]
        LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,
    )
else:
    from .attention_processor_faceid import (
        LoRAIPAttnProcessor,
    )


class FacePerceiverResampler(torch.nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        *,
        dim: int = 768,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        embedding_dim: int = 1280,
        output_dim: int = 768,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class MLPProjModel(torch.nn.Module):
    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class ProjPlusModel(torch.nn.Module):
    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        clip_embeddings_dim: int = 1280,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(
        self,
        id_embeds: torch.Tensor,
        clip_embeds: torch.Tensor,
        shortcut: bool = False,
        scale: float = 1.0,
    ) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class IPAdapterFaceID:
    def __init__(  # noqa: PLR0913
        self,
        sd_pipe: torch.nn.Module,
        ip_ckpt: str,
        device: torch.device,
        num_tokens: int = 4,
        n_cond: int = 1,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.n_cond = n_cond
        self.torch_dtype = torch_dtype

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self) -> torch.nn.Module:
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.torch_dtype)
        return image_proj_model

    def set_ip_adapter(self) -> None:
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens * self.n_cond,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self) -> None:
        if Path(self.ip_ckpt).suffix == ".safetensors":
            state_dict: dict[str, Any] = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = (
                            f.get_tensor(key)
                        )
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                            f.get_tensor(key)
                        )
        else:
            state_dict = loadfile(self.ip_ckpt, device="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(
        self, faceid_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multi_face = False
        if faceid_embeds.dim() == 3:  # noqa: PLR2004
            multi_face = True
            b, n, c = faceid_embeds.shape
            faceid_embeds = faceid_embeds.reshape(b * n, c)

        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_embeds = self.image_proj_model(faceid_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(faceid_embeds)
        )
        if multi_face:
            c = image_prompt_embeds.size(-1)
            image_prompt_embeds = image_prompt_embeds.reshape(b, -1, c)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape(b, -1, c)

        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale: float) -> None:
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(  # type: ignore[override] # noqa: PLR0913
        self,
        faceid_embeds: torch.Tensor | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        scale: float = 1.0,
        num_samples: int = 4,
        seed: int | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **_kwargs: object,
    ) -> list[Image.Image]:
        self.set_scale(scale)

        if faceid_embeds is None:
            raise ValueError("faceid_embeds is required")

        num_prompts = faceid_embeds.size(0)

        raw_prompt = prompt if prompt is not None else "best quality, high quality"
        raw_neg = (
            negative_prompt
            if negative_prompt is not None
            else "monochrome, lowres, bad anatomy, worst quality, low quality"
        )

        batch_prompts: list[str]
        if isinstance(raw_prompt, list):
            batch_prompts = raw_prompt
        else:
            batch_prompts = [raw_prompt] * num_prompts

        batch_neg: list[str]
        if isinstance(raw_neg, list):
            batch_neg = raw_neg
        else:
            batch_neg = [raw_neg] * num_prompts

        if not isinstance(prompt, list):
            faceid_embeds = faceid_embeds.repeat(num_samples, 1, 1)
            actual_num_samples = 1
        else:
            actual_num_samples = num_samples

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            faceid_embeds
        )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, actual_num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * actual_num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, actual_num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * actual_num_samples, seq_len, -1
        )

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                batch_prompts,
                device=self.device,
                num_images_per_prompt=actual_num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=batch_neg,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            **_kwargs,
        ).images

        return images


class IPAdapterFaceIDPlus:
    def __init__(  # noqa: PLR0913
        self,
        sd_pipe: torch.nn.Module,
        image_encoder_path: str,
        ip_ckpt: str,
        device: torch.device,
        num_tokens: int = 4,
        lora_rank: int = 128,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.lora_rank = lora_rank
        self.torch_dtype = torch_dtype

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_path
        ).to(self.device, dtype=self.torch_dtype)
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self) -> torch.nn.Module:
        image_proj_model = ProjPlusModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.torch_dtype)
        return image_proj_model

    def set_ip_adapter(self) -> None:
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self) -> None:
        if Path(self.ip_ckpt).suffix == ".safetensors":
            state_dict: dict[str, Any] = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = (
                            f.get_tensor(key)
                        )
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                            f.get_tensor(key)
                        )
        else:
            state_dict = loadfile(self.ip_ckpt, device="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(
        self,
        faceid_embeds: torch.Tensor,
        face_image: Image.Image | list[Image.Image],
        s_scale: float,
        shortcut: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle the image type explicitly for MyPy compatibility
        pil_image_list: list[Image.Image]
        if isinstance(face_image, Image.Image):
            pil_image_list = [face_image]
        else:
            pil_image_list = face_image

        clip_image = self.clip_image_processor(
            images=pil_image_list, return_tensors="pt"
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=self.torch_dtype)
        clip_image_embeds = self.image_encoder(
            clip_image, output_hidden_states=True
        ).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]

        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_embeds = self.image_proj_model(
            faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale
        )
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(faceid_embeds),
            uncond_clip_image_embeds,
            shortcut=shortcut,
            scale=s_scale,
        )
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale: float) -> None:
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale

    def generate(  # type: ignore[override] # noqa: PLR0913
        self,
        face_image: Image.Image | list[Image.Image] | None = None,
        faceid_embeds: torch.Tensor | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        scale: float = 1.0,
        num_samples: int = 4,
        seed: int | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        s_scale: float = 1.0,
        shortcut: bool = False,
        **_kwargs: object,
    ) -> list[Image.Image]:
        self.set_scale(scale)

        if faceid_embeds is None:
            raise ValueError("faceid_embeds required")
        if face_image is None:
            raise ValueError("face_image required")

        num_prompts = faceid_embeds.size(0)

        raw_prompt = prompt if prompt is not None else "best quality, high quality"
        raw_neg = (
            negative_prompt
            if negative_prompt is not None
            else "monochrome, lowres, bad anatomy, worst quality, low quality"
        )

        batch_prompts: list[str]
        if isinstance(raw_prompt, list):
            batch_prompts = raw_prompt
        else:
            batch_prompts = [raw_prompt] * num_prompts

        batch_neg: list[str]
        if isinstance(raw_neg, list):
            batch_neg = raw_neg
        else:
            batch_neg = [raw_neg] * num_prompts

        if not isinstance(prompt, list):
            faceid_embeds = faceid_embeds.repeat(num_samples, 1, 1)
            if isinstance(face_image, list):
                face_image = face_image * num_samples
            actual_num_samples = 1
        else:
            actual_num_samples = num_samples

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            faceid_embeds, face_image, s_scale, shortcut
        )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, actual_num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * actual_num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, actual_num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * actual_num_samples, seq_len, -1
        )

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                batch_prompts,
                device=self.device,
                num_images_per_prompt=actual_num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=batch_neg,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            **_kwargs,
        ).images

        return images


class IPAdapterFaceIDPlusXL(IPAdapterFaceIDPlus):
    """SDXL"""

    def generate(  # type: ignore[override] # noqa: PLR0913
        self,
        face_image: Image.Image | list[Image.Image] | None = None,
        faceid_embeds: torch.Tensor | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        scale: float = 1.0,
        num_samples: int = 4,
        seed: int | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        s_scale: float = 1.0,
        shortcut: bool = True,
        **_kwargs: object,
    ) -> list[Image.Image]:
        self.set_scale(scale)

        if faceid_embeds is None:
            raise ValueError("faceid_embeds required")
        if face_image is None:
            raise ValueError("face_image required")

        num_prompts = faceid_embeds.size(0)

        raw_prompt = prompt if prompt is not None else "best quality, high quality"
        raw_neg = (
            negative_prompt
            if negative_prompt is not None
            else "monochrome, lowres, bad anatomy, worst quality, low quality"
        )

        batch_prompts: list[str]
        if isinstance(raw_prompt, list):
            batch_prompts = raw_prompt
        else:
            batch_prompts = [raw_prompt] * num_prompts

        batch_neg: list[str]
        if isinstance(raw_neg, list):
            batch_neg = raw_neg
        else:
            batch_neg = [raw_neg] * num_prompts

        if not isinstance(prompt, list):
            faceid_embeds = faceid_embeds.repeat(num_samples, 1, 1)
            if isinstance(face_image, list):
                face_image = face_image * num_samples
            actual_num_samples = 1
        else:
            actual_num_samples = num_samples

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            faceid_embeds, face_image, s_scale, shortcut
        )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, actual_num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * actual_num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, actual_num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * actual_num_samples, seq_len, -1
        )

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                batch_prompts,
                device=self.device,
                num_images_per_prompt=actual_num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=batch_neg,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            **_kwargs,
        ).images

        return images
