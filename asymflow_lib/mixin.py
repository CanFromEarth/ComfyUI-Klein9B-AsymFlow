# Copyright (c) 2026 Hansheng Chen
# From: https://github.com/Lakonik/LakonLab
# Simplified: only AsymFlux2 adapter support, no GMFlow/other models.

import os
from typing import Union, Optional

import torch
import accelerate
import diffusers
from diffusers.models import AutoModel
from diffusers.models.modeling_utils import (
    load_state_dict,
    _LOW_CPU_MEM_USAGE_DEFAULT,
    no_init_weights,
    ContextManagers
)
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.utils.torch_utils import empty_device_cache
from .asymflux2_model import AsymFlux2Transformer2DModel

LOCAL_CLASS_MAPPING = {
    "AsymFlux2Transformer2DModel": AsymFlux2Transformer2DModel,
}

_SET_ADAPTER_SCALE_FN_MAPPING.update(
    AsymFlux2Transformer2DModel=lambda model_cls, weights: weights,
)

logger = logging.get_logger(__name__)


def assign_param(module, tensor_name: str, param: torch.nn.Parameter):
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    module._parameters[tensor_name] = param


class LakonLabMixin:

    def load_lakonlab_adapter(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_module_name: str = "transformer",
        adapter_name: Optional[str] = None,
        **kwargs
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        disable_mmap = kwargs.pop("disable_mmap", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        user_agent = {
            "diffusers": diffusers.__version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = AutoModel.load_config(pretrained_model_name_or_path, subfolder=subfolder, **load_config_kwargs)

        orig_class_name = config["_class_name"]

        if orig_class_name in LOCAL_CLASS_MAPPING:
            model_cls = LOCAL_CLASS_MAPPING[orig_class_name]
        else:
            load_config_kwargs.update({"subfolder": subfolder})
            from diffusers.pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates
            model_cls, _ = get_class_obj_and_candidates(
                library_name="diffusers",
                class_name=orig_class_name,
                importable_classes=ALL_IMPORTABLE_CLASSES,
                pipelines=None,
                is_pipeline_module=False,
            )

        if model_cls is None:
            raise ValueError(f"Can't find a model linked to {orig_class_name}.")

        # Get model file
        model_file = None

        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
            except IOError as e:
                logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                if not allow_pickle:
                    raise
                logger.warning(
                    "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                )

        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )

        assert model_file is not None, \
            f"Could not find adapter weights for {pretrained_model_name_or_path}."

        # Initialize model
        base_module = getattr(self, target_module_name)

        torch_dtype = base_module.dtype
        device = base_module.device
        dtype_orig = model_cls._set_default_torch_dtype(torch_dtype)

        overwrite_state_dict = dict()
        lora_state_dict = dict()

        adapter_state_dict = load_state_dict(model_file, disable_mmap=disable_mmap)
        for k in adapter_state_dict.keys():
            adapter_state_dict[k] = adapter_state_dict[k].to(dtype=torch_dtype, device=device)
            if "lora" in k:
                lora_state_dict[k.removeprefix(f"{target_module_name}.")] = adapter_state_dict[k]
            else:
                overwrite_state_dict[k.removeprefix(f"{target_module_name}.")] = adapter_state_dict[k]

        # Determine initial quantization config
        pre_quantized = ("quantization_config" in base_module.config
                         and base_module.config["quantization_config"] is not None)
        if pre_quantized:
            config["quantization_config"] = base_module.config.quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(
                config["quantization_config"], pre_quantized=True
            )
            hf_quantizer.validate_environment(torch_dtype=torch_dtype)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")
        else:
            hf_quantizer = None

        use_keep_in_fp32_modules = model_cls._keep_in_fp32_modules is not None and (
            hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        )

        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = model_cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        else:
            keep_in_fp32_modules = []

        for k in overwrite_state_dict.keys():
            module_name = k.rsplit('.', 1)[0]
            if module_name and module_name not in keep_in_fp32_modules:
                keep_in_fp32_modules.append(module_name)

        init_contexts = [no_init_weights()]

        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())

        with ContextManagers(init_contexts):
            piflow_module = model_cls.from_config(config).eval()

        torch.set_default_dtype(dtype_orig)

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=piflow_module, device_map=None, keep_in_fp32_modules=keep_in_fp32_modules
            )

        # Load model weights
        base_state_dict = base_module.state_dict()
        base_state_dict.update(overwrite_state_dict)
        empty_state_dict = piflow_module.state_dict()
        for param_name, param in base_state_dict.items():
            if param_name not in empty_state_dict:
                continue
            if hf_quantizer is not None and (
                    hf_quantizer.check_if_quantized_param(
                        piflow_module, param, param_name, base_state_dict, param_device=device)):
                hf_quantizer.create_quantized_param(
                    piflow_module, param, param_name, device, base_state_dict, unexpected_keys=[], dtype=torch_dtype
                )
            else:
                assign_param(piflow_module, param_name, param)

        empty_device_cache()

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(piflow_module)
            piflow_module.hf_quantizer = hf_quantizer

        if len(lora_state_dict) == 0:
            adapter_name = None
        else:
            if adapter_name is None:
                adapter_name = f"{target_module_name}_piflow"
            piflow_module.load_lora_adapter(
                lora_state_dict, prefix=None, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        if adapter_name is None:
            logger.warning(
                f"No LoRA weights were found in {pretrained_model_name_or_path}."
            )

        setattr(self, target_module_name, piflow_module)

        return adapter_name
