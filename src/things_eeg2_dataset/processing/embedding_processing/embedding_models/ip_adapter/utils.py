from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

attn_maps = {}


def hook_fn(name: str) -> Callable:
    def forward_hook(module: torch.nn.Module, _: tuple, output: torch.Tensor) -> None:
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet: torch.nn.Module) -> torch.nn.Module:
    for name, module in unet.named_modules():
        if name.split(".")[-1].startswith("attn2"):
            module.register_forward_hook(hook_fn(name))

    return unet


def upscale(attn_map: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1, 0)
    temp_size = None

    for i in range(0, 5):
        scale = 2**i
        if (target_size[0] // scale) * (target_size[1] // scale) == attn_map.shape[
            1
        ] * 64:
            temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
            break

    if temp_size is None:
        raise ValueError("temp_size cannot be None")

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map


def get_net_attn_map(
    image_size: tuple[int, int],
    batch_size: int = 2,
    instance_or_negative: bool = False,
    detach: bool = True,
) -> torch.Tensor:
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for attn_map in attn_maps.values():
        attn_map_processed = attn_map.cpu() if detach else attn_map
        attn_map_processed = torch.chunk(attn_map_processed, batch_size)[idx].squeeze()
        attn_map_processed = upscale(attn_map, image_size)
        net_attn_maps.append(attn_map)

    net_attn_maps = torch.mean(torch.stack(net_attn_maps, dim=0), dim=0)

    return net_attn_maps


def attnmaps2images(net_attn_maps: torch.Tensor) -> list[Image.Image]:
    # total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        _attn_map = attn_map.cpu().numpy()
        # total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (
            (_attn_map - np.min(_attn_map))
            / (np.max(_attn_map) - np.min(_attn_map))
            * 255
        )
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        # print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        # image = fix_save_attn_map(attn_map)
        images.append(image)

    # print(total_attn_scores)
    return images


def is_torch2_available() -> bool:
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(
    seed: int | list[int] | None, device: torch.device
) -> torch.Generator | list[torch.Generator] | None:
    if seed is not None:
        if isinstance(seed, list):
            generator = [
                torch.Generator(device).manual_seed(seed_item) for seed_item in seed
            ]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
