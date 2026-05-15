"""
AsymFLUX.2 Klein -- Pixel-space text-to-image generation for ComfyUI.
Based on AsymFlow by Hansheng Chen et al. (Stanford University)
https://github.com/Lakonik/LakonLab
"""

import os
import math
import logging

import torch
import numpy as np
import folder_paths

logger = logging.getLogger("[AsymFlow]")

# Global pipeline cache
_pipe_cache = {}

# Register diffusers model folder
_DIFFUSERS_DIR = os.path.join(folder_paths.models_dir, "diffusers")
os.makedirs(_DIFFUSERS_DIR, exist_ok=True)


def _list_diffusers_models():
    if not os.path.isdir(_DIFFUSERS_DIR):
        return []
    return sorted(
        d for d in os.listdir(_DIFFUSERS_DIR)
        if os.path.isdir(os.path.join(_DIFFUSERS_DIR, d))
    )


def _resolve_model_path(name: str) -> str:
    if os.path.isabs(name) and os.path.isdir(name):
        return name
    candidate = os.path.join(_DIFFUSERS_DIR, name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Model not found: '{name}'. "
        f"Please download it to {_DIFFUSERS_DIR}/{name}\n"
        f"  huggingface-cli download <repo_id> --local-dir {candidate}"
    )


def _get_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class AsymFlux2KleinLoader:
    """Load the AsymFLUX.2 klein pixel pipeline from local model files."""

    @classmethod
    def INPUT_TYPES(cls):
        available = _list_diffusers_models()
        return {
            "required": {
                "base_model": (
                    available if available else ["FLUX.2-klein-base-9B"],
                    {},
                ),
                "adapter": (
                    available if available else ["AsymFLUX.2-klein-9B"],
                    {},
                ),
            },
            "optional": {
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {"default": "bfloat16"},
                ),
                "device": (["cuda", "mps", "cpu"], {"default": "cuda"}),
                "enable_cpu_offload": (
                    "BOOLEAN",
                    {"default": False, "label_on": "True", "label_off": "False"},
                ),
            },
        }

    RETURN_TYPES = ("ASYMFLUX_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "AsymFlow"
    DESCRIPTION = (
        "Load the AsymFLUX.2 klein 9B pixel-space pipeline.\n"
        "Place models in ComfyUI/models/diffusers/."
    )

    def load(
        self,
        base_model,
        adapter,
        dtype="bfloat16",
        device="cuda",
        enable_cpu_offload=False,
    ):
        base_path = _resolve_model_path(base_model)
        adapter_path = _resolve_model_path(adapter)

        cache_key = (base_path, adapter_path, dtype, device, enable_cpu_offload)
        if cache_key in _pipe_cache:
            logger.info("Using cached AsymFLUX.2 klein pipeline")
            return (_pipe_cache[cache_key],)

        from .asymflow_lib import (
            PixelFlux2KleinPipeline,
            OklabColorEncoder,
            FlowAdapterScheduler,
        )

        logger.info(f"Loading AsymFLUX.2 klein pipeline from: {base_path}")
        torch_dtype = _get_dtype(dtype)

        pipe = PixelFlux2KleinPipeline.from_pretrained(
            base_path,
            vae=OklabColorEncoder(
                use_affine_norm=True,
                mean=(0.56, 0.0, 0.01),
                std=0.16,
            ),
            scheduler=FlowAdapterScheduler(
                shift=17.0,
                use_dynamic_shifting=True,
                base_seq_len=1024**2,
                max_seq_len=2048**2,
                base_logshift=math.log(17.0),
                max_logshift=math.log(34.0),
                dynamic_shifting_type="sqrt",
                base_scheduler="UniPCMultistep",
            ),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

        logger.info(f"Loading adapter from: {adapter_path}")
        pipe.load_lakonlab_adapter(adapter_path, target_module_name="transformer")

        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        _pipe_cache[cache_key] = pipe
        logger.info("AsymFLUX.2 klein pipeline ready")
        return (pipe,)


class AsymFlux2KleinSampler:
    """Generate pixel-space images with AsymFLUX.2 klein."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ASYMFLUX_PIPE",),
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 2}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Low quality, worst quality, blurry, deformed, bad anatomy",
                    },
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 16},
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 38, "min": 1, "max": 150},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "orthogonal_guidance": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "clamp_denoised": (
                    "BOOLEAN",
                    {"default": True, "label_on": "True", "label_off": "False"},
                ),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "AsymFlow"
    DESCRIPTION = "Generate pixel-space images using AsymFLUX.2 klein. Supports text-to-image and image-to-image."

    def generate(
        self,
        pipe,
        prompt,
        seed,
        negative_prompt="Low quality, worst quality, blurry, deformed, bad anatomy",
        width=1024,
        height=1024,
        num_inference_steps=38,
        guidance_scale=4.0,
        orthogonal_guidance=1.0,
        clamp_denoised=True,
        image=None,
    ):
        from PIL import Image

        input_image = None
        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            input_image = Image.fromarray(img_np)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=input_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            orthogonal_guidance=orthogonal_guidance,
            clamp_denoised=clamp_denoised,
            generator=generator,
        )

        pil_image = result.images[0]
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return (img_tensor,)
