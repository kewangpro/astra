"""
SpatialAnalyzer — Step 3.5.

Generates CNN saliency / activation maps for visual environments (Snake, Tetris).
Exposed via the API so the Model Registry deep-dive view can render them.

Requires: torch, torchvision (installed in the sandbox, imported lazily).
"""
from __future__ import annotations

import os
from typing import Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


class SpatialAnalyzer:
    """
    Computes gradient-weighted class activation maps (Grad-CAM) for CNN policies.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _load_model(self):
        try:
            import torch
            self._model = torch.load(self.checkpoint_path, map_location="cpu")
            self._model.eval()
        except Exception as e:
            logger.error("SpatialAnalyzer: failed to load model: %s", e)
            raise

    def generate_saliency_map(self, observation, layer_name: Optional[str] = None) -> dict:
        """
        Compute a Grad-CAM saliency map for the given observation tensor.
        Returns a dict with the raw activation map and metadata.
        """
        if self._model is None:
            self._load_model()

        try:
            import torch
            import torch.nn.functional as F

            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            activations = {}
            gradients = {}

            # Register hooks on the target layer
            target_layer = self._get_target_layer(layer_name)
            if target_layer is None:
                return {"error": "target layer not found", "map": None}

            def save_activation(module, inp, out):
                activations["value"] = out.detach()

            def save_gradient(module, grad_in, grad_out):
                gradients["value"] = grad_out[0].detach()

            fwd_hook = target_layer.register_forward_hook(save_activation)
            bwd_hook = target_layer.register_full_backward_hook(save_gradient)

            output = self._model(obs_tensor)
            score = output.max()
            score.backward()

            fwd_hook.remove()
            bwd_hook.remove()

            # Compute weighted activation map
            weights = gradients["value"].mean(dim=[2, 3], keepdim=True)
            cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
            cam = cam.squeeze().numpy().tolist()

            return {"layer": layer_name or "auto", "map": cam, "shape": str(obs_tensor.shape)}

        except Exception as e:
            logger.error("SpatialAnalyzer: Grad-CAM failed: %s", e)
            return {"error": str(e), "map": None}

    def _get_target_layer(self, layer_name: Optional[str]):
        if self._model is None:
            return None
        if layer_name:
            return dict(self._model.named_modules()).get(layer_name)
        # Auto-select last Conv2d layer
        last_conv = None
        try:
            import torch.nn as nn
            for module in self._model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
        except ImportError:
            pass
        return last_conv
