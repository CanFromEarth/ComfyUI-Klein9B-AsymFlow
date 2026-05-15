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

# AsymFLUX.2 Klein 9B adapter config (from Lakonik/AsymFLUX.2-klein-9B)
_ASYMFLUX2_KLEIN_CONFIG = {
    "patch_size": 16,
    "in_channels": 3,
    "base_rank": 128,
    "num_layers": 8,
    "num_single_layers": 24,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "joint_attention_dim": 12288,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "sigma_min": 1e-4,
    "num_timesteps": 1,
    "guidance_embeds": False,
}


def _get_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _list_model_dirs(folder_name):
    """List subdirectories in a ComfyUI model folder."""
    dirs = []
    try:
        paths = folder_paths.get_folder_paths(folder_name)
    except KeyError:
        return dirs
    for base_path in paths:
        if not os.path.isdir(base_path):
            continue
        for d in sorted(os.listdir(base_path)):
            if os.path.isdir(os.path.join(base_path, d)):
                dirs.append(d)
    return dirs


def _resolve_model_dir(folder_name, dir_name):
    """Get full path for a subdirectory in a ComfyUI model folder."""
    try:
        paths = folder_paths.get_folder_paths(folder_name)
    except KeyError:
        paths = []
    for base_path in paths:
        full = os.path.join(base_path, dir_name)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(
        f"Directory '{dir_name}' not found in {folder_name}/ model folder."
    )


class AsymFlux2KleinLoader:
    """Load the AsymFLUX.2 klein pixel pipeline from local model files.

    - Transformer .safetensors from models/diffusion_models/
    - Text encoder directory from models/text_encoders/ or models/clip/
    - Adapter .safetensors from models/loras/
    """

    @classmethod
    def INPUT_TYPES(cls):
        diff_models = folder_paths.get_filename_list("diffusion_models")
        te_dirs = sorted(set(
            _list_model_dirs("text_encoders") + _list_model_dirs("clip")
        ))
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "transformer": (diff_models, {}),
                "text_encoder": (te_dirs if te_dirs else ["(place model dir in text_encoders/ or clip/)"], {}),
                "adapter": (loras, {}),
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
        "Transformer: models/diffusion_models/ (.safetensors)\n"
        "Text encoder: models/text_encoders/ or models/clip/ (directory)\n"
        "Adapter: models/loras/ (.safetensors)"
    )

    def load(
        self,
        transformer,
        text_encoder,
        adapter,
        dtype="bfloat16",
        device="cuda",
        enable_cpu_offload=False,
    ):
        transformer_path = folder_paths.get_full_path("diffusion_models", transformer)
        adapter_path = folder_paths.get_full_path("loras", adapter)
        # Look in both text_encoders/ and clip/
        try:
            te_dir = _resolve_model_dir("text_encoders", text_encoder)
        except FileNotFoundError:
            te_dir = _resolve_model_dir("clip", text_encoder)

        cache_key = (transformer_path, te_dir, adapter_path, dtype, device, enable_cpu_offload)
        if cache_key in _pipe_cache:
            logger.info("Using cached AsymFLUX.2 klein pipeline")
            return (_pipe_cache[cache_key],)

        torch_dtype = _get_dtype(dtype)

        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
        from .asymflow_lib import (
            PixelFlux2KleinPipeline,
            OklabColorEncoder,
            FlowAdapterScheduler,
        )
        from .asymflow_lib.asymflux2_model import AsymFlux2Transformer2DModel

        # 1. Load base transformer weights
        logger.info(f"Loading transformer weights: {transformer_path}")
        base_state_dict = load_file(transformer_path, device="cpu")

        # Strip "transformer." prefix if present (consolidated checkpoints)
        if any(k.startswith("transformer.") for k in list(base_state_dict.keys())[:5]):
            base_state_dict = {
                k.removeprefix("transformer."): v
                for k, v in base_state_dict.items()
                if k.startswith("transformer.")
            }

        # 2. Load adapter weights and split into overwrites + LoRA
        logger.info(f"Loading adapter weights: {adapter_path}")
        adapter_state_dict = load_file(adapter_path, device="cpu")

        overwrite_state_dict = {}
        lora_state_dict = {}
        for k, v in adapter_state_dict.items():
            k_clean = k.removeprefix("transformer.")
            if "lora" in k_clean:
                lora_state_dict[k_clean] = v.to(dtype=torch_dtype)
            else:
                overwrite_state_dict[k_clean] = v.to(dtype=torch_dtype)
        del adapter_state_dict

        # 3. Merge: base weights + adapter overwrites
        for k in base_state_dict:
            base_state_dict[k] = base_state_dict[k].to(dtype=torch_dtype)
        base_state_dict.update(overwrite_state_dict)
        del overwrite_state_dict

        # 4. Create AsymFlux2 model and load merged weights
        logger.info("Creating AsymFlux2 transformer model")
        with init_empty_weights():
            transformer_model = AsymFlux2Transformer2DModel(**_ASYMFLUX2_KLEIN_CONFIG)

        transformer_model.load_state_dict(base_state_dict, strict=False, assign=True)
        del base_state_dict

        # 5. Load LoRA weights
        if lora_state_dict:
            logger.info("Loading LoRA adapter weights")
            transformer_model.load_lora_adapter(
                lora_state_dict, prefix=None, adapter_name="asymflow", low_cpu_mem_usage=True
            )
        del lora_state_dict

        # 6. Load text encoder + tokenizer
        logger.info(f"Loading text encoder: {te_dir}")

        # Support both flat layout and BFL-style subdirectories
        te_subdir = os.path.join(te_dir, "text_encoder")
        tok_subdir = os.path.join(te_dir, "tokenizer")

        te_load_path = te_subdir if os.path.isdir(te_subdir) else te_dir
        tok_load_path = tok_subdir if os.path.isdir(tok_subdir) else te_dir

        text_encoder_model = Qwen3ForCausalLM.from_pretrained(
            te_load_path, torch_dtype=torch_dtype, local_files_only=True
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            tok_load_path, local_files_only=True
        )

        # 7. Construct pipeline
        logger.info("Constructing AsymFLUX.2 klein pipeline")
        pipe = PixelFlux2KleinPipeline(
            transformer=transformer_model,
            text_encoder=text_encoder_model,
            tokenizer=tokenizer,
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
        )

        # 8. Move to device
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
