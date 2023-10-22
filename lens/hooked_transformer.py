import argparse
import functools
import torch
import torch.nn as nn
import transformers

from collections import OrderedDict
from functools import partial
from PIL.Image import Image
from typing import List, Optional, Dict
from transformers import (
    AutoConfig,
    CLIPConfig,
    CLIPModel,
    ViTConfig, 
    ViTForMaskedImageModeling, 
    PretrainedConfig, 
    AutoModel
)

from transformer_lens.hook_points import HookPoint
from .hook_points import HFHookedRootModule


class HookedVisionTransformer(HFHookedRootModule):
    """
    This class implements an interface to VisionTransformer implementations from HuggingFace, with HookPoints on all interesting activations.
    
    It can instantiated using pretrained weights via the HookedVisionTransformer.from_pretrained class method, or with randomly initialized 
    weights via HookedVisionTransformer.from_config class method.
    """

    def __init__(self, model_name_or_path: str, config: PretrainedConfig, device: torch.device):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.config = config

        # check if model has been instantiated with either config or weights, if not throw an error
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"{self.__class__.__name__} is designed to be instantiated "
                f"using `{self.__class__.__name__}.from_pretrained(model_name_or_path)` "
                f"or using `{self.__class__.__name__}.from_config(config)`."
            )

        # setup model based on config or weights
        self.model = AutoModel.from_pretrained(self.model_name_or_path, config = self.config)
        self.model.to(device)
        self.model.eval()

        # inject hook points into model implementation, build a hook directory mapping module names to module instances, and another directory 
        # that stores references to hook points
        self.mod_dict = {}
        self.hook_dict: Dict[str, HookPoint] = {}
        self.add_hook_points()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: torch.device):

        # instantiate class with model configuration and weights
        config = AutoConfig.from_pretrained(model_name_or_path)
        cls_instance = cls(
            model_name_or_path=model_name_or_path,
            config=config,
            device=device
        )
        return cls_instance

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
