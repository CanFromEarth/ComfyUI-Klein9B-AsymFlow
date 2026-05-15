# ComfyUI-AsymFlow

Standalone ComfyUI node pack for [AsymFLUX.2 Klein](https://hanshengchen.com/asymflow) -- pixel-space text-to-image generation.

Based on [AsymFlow: Asymmetric Flow Models](https://arxiv.org/abs/2605.12964) by Hansheng Chen et al. (Stanford University).
Core inference code extracted from [LakonLab](https://github.com/Lakonik/LakonLab) and bundled for self-contained use -- no need to clone the full LakonLab repository.

## Nodes

| Node | Purpose |
|------|---------|
| **AsymFLUX.2 Klein Loader** | Load base model + AsymFlow adapter into a pixel-space pipeline |
| **AsymFLUX.2 Klein Sampler** | Text-to-image and image-to-image generation |

## Setup

### 1. Install dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-AsymFlow
pip install -r requirements.txt
```

### 2. Download models

Place models in `ComfyUI/models/diffusers/`:

```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-base-9B \
    --local-dir ComfyUI/models/diffusers/FLUX.2-klein-base-9B

huggingface-cli download Lakonik/AsymFLUX.2-klein-9B \
    --local-dir ComfyUI/models/diffusers/AsymFLUX.2-klein-9B
```

### 3. Use in ComfyUI

1. Add **AsymFLUX.2 Klein Loader** node, select base model and adapter
2. Connect its output to **AsymFLUX.2 Klein Sampler**
3. Enter a prompt and generate

## Recommended Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| Steps | 38 | |
| Guidance Scale | 4.0 | |
| Orthogonal Guidance | 1.0 | Controls CFG orthogonality |
| Clamp Denoised | True | Improves color accuracy |
| dtype | bfloat16 | Use float16 if bfloat16 unsupported |

## Credits

- **AsymFlow** paper and code: Hansheng Chen, Jan Ackermann, Minseo Kim, Gordon Wetzstein, Leonidas Guibas (Stanford University)
- **FLUX.2 Klein**: Black Forest Labs

## License

The bundled inference code in `asymflow_lib/` is derived from [LakonLab](https://github.com/Lakonik/LakonLab). Please refer to the original repository for licensing terms.
