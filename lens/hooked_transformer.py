import argparse

import torch
import torch.nn as nn
import transformers

from functools import partial
from PIL.Image import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor, PretrainedConfig, AutoConfig
from typing import List, Optional, Dict

from transformer_lens.hook_points import HookedRootModule, HookPoint

import functools


## bits and bytes

class HookedVisionTransformer(HookedRootModule):
    """
    This class implements an interface to VisionTransformer implementations from HuggingFace, with HookPoints on all interesting activations.
    
    It can instantiated using pretrained weights via the HookedVisionTransformer.from_pretrained class method, or with randomly initialized 
    weights via HookedVisionTransformer.from_config class method.
    """

    def __init__(self, model_name_or_path: str, config: PretrainedConfig, device: torch.device) -> None:
        super().__init__()

        # check if model has been instantiated with either config or weights, if not throw an error
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"{self.__class__.__name__} is designed to be instantiated "
                f"using `{self.__class__.__name__}.from_pretrained(model_name_or_path)` "
                f"or using `{self.__class__.__name__}.from_config(config)`."
            )

        self.model_name_or_path = model_name_or_path
        self.config = config

        # setup model based on config and weights
        self.model = ViTForImageClassification.from_pretrained(self.model_name_or_path, config=self.config)
        self.model.to(device)
        self.model.eval()

        # inject hook points into model implementation, build a hook directory mapping module names to
        # module instances, and another directory that stores references to hook points
        self.mod_dict = {}
        self.hook_dict: Dict[str, HookPoint] = {}
        self.inject_hook_points()

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

    def add_hook_point(self, func, description):
        # a decorator to wrap a function into a HookPoints

        hook_point = HookPoint()
        hook_point.name = description
        self.mod_dict[description] = hook_point
        self.hook_dict[description] = hook_point

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return hook_point(func(self, *args, **kwargs))
        return wrapper

    def inject_hook_points(self):
        """post-hoc injection of hook points"""

        hook_points = self.select_hook_points()

        # inject the HookPoints using the decorator
        for description, (path, func) in hook_points.items():
            wrapped_func = self.add_hook_point(func, description)
            # Update the original method with the wrapped function
            parts = path.split('.')
            target = self.model
            for i, part in enumerate(parts[:-1]):
                if "[" in part and "]" in part:
                    # Handle the case where the part is an index like 'layer[0]'
                    attribute_name, index_str = part.split('[')
                    index = int(index_str.rstrip(']'))
                    target = getattr(target, attribute_name)[index]
                else:
                    target = getattr(target, part)
            setattr(target, parts[-1], wrapped_func)


    def select_hook_points(self):

        hook_points = {}
        
        hook_points["embeddings"] = ("vit.embeddings.forward", self.model.vit.embeddings.forward)
        for idx, layer in enumerate(self.model.vit.encoder.layer):
            hook_points[f"encoder.{idx}.ln1"] = (f"vit.encoder.layer[{idx}].layernorm_before.forward", layer.layernorm_before.forward)
            hook_points[f"encoder.{idx}.attn.q"] = (f"vit.encoder.layer[{idx}].attention.attention.query.forward", layer.attention.attention.query.forward)
            hook_points[f"encoder.{idx}.attn.k"] = (f"vit.encoder.layer[{idx}].attention.attention.key.forward", layer.attention.attention.key.forward)
            hook_points[f"encoder.{idx}.attn.v"] = (f"vit.encoder.layer[{idx}].attention.attention.value.forward", layer.attention.attention.value.forward)
            hook_points[f"encoder.{idx}.ln2"] = (f"vit.encoder.layer[{idx}].layernorm_after.forward", layer.layernorm_after.forward)
            hook_points[f"encoder.{idx}.intermediate"] = (f"vit.encoder.layer[{idx}].intermediate.forward", layer.intermediate.forward)
        hook_points["ln"] = ("vit.layernorm.forward", self.model.vit.layernorm.forward)
        return hook_points




