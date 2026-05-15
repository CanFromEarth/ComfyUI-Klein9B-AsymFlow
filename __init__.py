from .nodes import AsymFlux2KleinLoader, AsymFlux2KleinSampler

NODE_CLASS_MAPPINGS = {
    "AsymFlux2KleinLoader": AsymFlux2KleinLoader,
    "AsymFlux2KleinSampler": AsymFlux2KleinSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymFlux2KleinLoader": "AsymFLUX.2 Klein Loader",
    "AsymFlux2KleinSampler": "AsymFLUX.2 Klein Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
