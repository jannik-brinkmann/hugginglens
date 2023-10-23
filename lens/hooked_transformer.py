import functools
import torch

from typing import Dict
from transformers import PretrainedConfig, PreTrainedModel
from transformer_lens.hook_points import HookPoint

from .hook_points import HFHookedRootModule


class HookedHFTransformer(HFHookedRootModule):
    """
    This class implements an interface to model implementations from HuggingFace, with HookPoints on all interesting activations.
    
    It can instantiated using pretrained weights via the HookedVisionTransformer.from_pretrained class method, or with randomly initialized 
    weights via HookedVisionTransformer.from_config class method.
    """

    def __init__(self, model: PreTrainedModel, device: torch.device = "cuda"):
        super().__init__()

        # setup model based on config or weights
        self.model = model
        self.model.to(device)
        self.model.eval()

        # inject hook points into model implementation, build a hook directory mapping module names to module instances, and another directory 
        # that stores references to hook points
        self.mod_dict = {}
        self.hook_dict: Dict[str, HookPoint] = {}
        self.add_hook_points()
        
    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs

    def add_hook_point(self, name, module):
        """a decorator to wrap a function into a HookPoint"""
        hook_point = HookPoint()
        hook_point.name = name
        self.mod_dict[name] = hook_point
        self.hook_dict[name] = hook_point

        @functools.wraps(module)
        def decorator(*args, **kwargs):
            return hook_point(module(*args, **kwargs))
        return decorator

    def add_hook_points(self):
        """post-hoc injection of hook points"""

        # determine named modules
        named_modules = {name: module for name, module in self.model.named_modules()}

        # inject a HookPoint at each named module using the decorator
        for name, module in named_modules.items():
            if name != '' and hasattr(module, "forward"):
                decorated_method = self.add_hook_point(name, getattr(module, "forward"))
                setattr(module, "forward", decorated_method)
