# ComfyUI-AsymFlow

<p align="center">
  <img src="assets/hero.png" alt="AsymFLUX.2 Klein sample outputs" />
</p>

Standalone ComfyUI node pack for [AsymFLUX.2 Klein](https://hanshengchen.com/asymflow) -- pixel-space text-to-image generation.

Based on [AsymFlow: Asymmetric Flow Models](https://arxiv.org/abs/2605.12964)
Core inference code extracted from [LakonLab](https://github.com/Lakonik/LakonLab).

## Nodes

| Node | Purpose |
|------|---------|
| **AsymFLUX.2 Klein Loader** | Load transformer + text encoder + adapter |
| **AsymFLUX.2 Klein Sampler** | Text-to-image and image-to-image generation |

## Setup

### 1. Install dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-AsymFlow
pip install -r requirements.txt
```

### 2. Download models

**Transformer** → `models/diffusion_models/`
```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-base-9B \
    flux-2-klein-base-9b.safetensors \
    --local-dir ComfyUI/models/diffusion_models/
```

**Text Encoder + Tokenizer** → `models/text_encoders/`
```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-base-9B \
    --include "text_encoder/*" "tokenizer/*" \
    --local-dir ComfyUI/models/text_encoders/FLUX2-klein-9B/
```

**Adapter** → `models/loras/`
```bash
huggingface-cli download Lakonik/AsymFLUX.2-klein-9B \
    diffusion_pytorch_model.safetensors \
    --local-dir ComfyUI/models/loras/
```

### 3. Use in ComfyUI

1. Add **AsymFLUX.2 Klein Loader** — select transformer, text encoder dir, and adapter
2. Connect to **AsymFLUX.2 Klein Sampler**
3. Enter a prompt and generate

## Model Locations

| Component | Folder | Format |
|-----------|--------|--------|
| Transformer | `models/diffusion_models/` | Single .safetensors |
| Text Encoder | `models/text_encoders/<name>/` | Directory (config.json + sharded .safetensors) |
| Adapter | `models/loras/` | Single .safetensors |

The text encoder directory supports both flat layout and BFL-style subdirectories (`text_encoder/` + `tokenizer/`).

## Recommended Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| Steps | 38 | |
| Guidance Scale | 4.0 | |
| Orthogonal Guidance | 1.0 | Controls CFG orthogonality |
| Clamp Denoised | True | Improves color accuracy |
| dtype | bfloat16 | Use float16 if bfloat16 unsupported |

## Credits

- **AsymFlow**: Hansheng Chen, Jan Ackermann, Minseo Kim, Gordon Wetzstein, Leonidas Guibas (Stanford University)
- **FLUX.2 Klein**: Black Forest Labs

## License

The bundled inference code in `asymflow_lib/` is derived from [LakonLab](https://github.com/Lakonik/LakonLab). Please refer to the original repository for licensing terms.
