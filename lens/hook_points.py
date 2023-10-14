from collections import OrderedDict
from transformers import ViTConfig


class BaseHookPoints:
    def __init__(self, model):
        self.model = model
        self.hook_points = {}
    def get_hook_points(self):
        raise NotImplementedError(f"'get_hook_points' not implemented for {self.model}.")


class ViTHookPoints(BaseHookPoints):
    def get_hook_points(self):
        for idx, layer in enumerate(self.model.vit.encoder.layer):
            self.hook_points[f"encoder.{idx}.ln1"] = (f"vit.encoder.layer[{idx}].layernorm_before.forward", layer.layernorm_before.forward)
            self.hook_points[f"encoder.{idx}.attn.q"] = (f"vit.encoder.layer[{idx}].attention.attention.query.forward", layer.attention.attention.query.forward)
            self.hook_points[f"encoder.{idx}.attn.k"] = (f"vit.encoder.layer[{idx}].attention.attention.key.forward", layer.attention.attention.key.forward)
            self.hook_points[f"encoder.{idx}.attn.v"] = (f"vit.encoder.layer[{idx}].attention.attention.value.forward", layer.attention.attention.value.forward)
            self.hook_points[f"encoder.{idx}.ln2"] = (f"vit.encoder.layer[{idx}].layernorm_after.forward", layer.layernorm_after.forward)
            self.hook_points[f"encoder.{idx}.intermediate"] = (f"vit.encoder.layer[{idx}].intermediate.forward", layer.intermediate.forward)
        self.hook_points["ln"] = ("vit.layernorm.forward", self.model.vit.layernorm.forward)
        return self.hook_points


HOOK_POINT_MAPPING = OrderedDict([
    (ViTConfig, ViTHookPoints)
])


def get_hook_points(model, config):

    if type(config) in HOOK_POINT_MAPPING.keys():
        cls = HOOK_POINT_MAPPING[type(config)](model)
    else:
        raise EnvironmentError(
            f'No HookPoints specified for {config}.'
        )
    return cls.get_hook_points()
