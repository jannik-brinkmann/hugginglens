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
    PretrainedConfig
)

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .hook_points import get_hook_points


MODEL_MAPPING = OrderedDict([
    (CLIPConfig, CLIPModel),
    (ViTConfig, ViTForMaskedImageModeling),
])


class HookedVisionTransformer(HookedRootModule):
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
        if type(config) in MODEL_MAPPING.keys():
            model_class = MODEL_MAPPING[type(config)]
        else:
            raise EnvironmentError(
                f'{self.__class__.__name__} is designed to be instantiated given a registered ModelConfig.'
            )
        self.model = model_class.from_pretrained(self.model_name_or_path, config=self.config)
        self.model.to(device)
        self.model.eval()

        # inject hook points into model implementation, build a hook directory mapping module names to
        # module instances, and another directory that stores references to hook points
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

    def add_hook_points(self):
        """post-hoc injection of hook points"""

        hook_points = get_hook_points(self.model, self.config)

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
