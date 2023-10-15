from collections import OrderedDict
from transformers import CLIPConfig, ViTConfig


class BaseHookPoints:
    def __init__(self):
        self.hook_points = {}
    def get_hook_points(self):
        raise NotImplementedError(f"'get_hook_points' not implemented for {self.model}.")


class ViTHookPoints(BaseHookPoints):
    def get_hook_points(self, model):
        self.hook_points["embeddings"] = ("vit.embeddings.forward", model.vit.embeddings.forward)
        for idx, layer in enumerate(model.vit.encoder.layer):
            self.hook_points[f"encoder.{idx}.ln1"] = (f"vit.encoder.layer[{idx}].layernorm_before.forward", layer.layernorm_before.forward)
            self.hook_points[f"encoder.{idx}.attn.q"] = (f"vit.encoder.layer[{idx}].attention.attention.query.forward", layer.attention.attention.query.forward)
            self.hook_points[f"encoder.{idx}.attn.k"] = (f"vit.encoder.layer[{idx}].attention.attention.key.forward", layer.attention.attention.key.forward)
            self.hook_points[f"encoder.{idx}.attn.v"] = (f"vit.encoder.layer[{idx}].attention.attention.value.forward", layer.attention.attention.value.forward)
            self.hook_points[f"encoder.{idx}.ln2"] = (f"vit.encoder.layer[{idx}].layernorm_after.forward", layer.layernorm_after.forward)
            self.hook_points[f"encoder.{idx}.intermediate"] = (f"vit.encoder.layer[{idx}].intermediate.forward", layer.intermediate.forward)
        self.hook_points["ln_final"] = ("vit.layernorm.forward", model.vit.layernorm.forward)
        return self.hook_points#


class CLIPHookPoints(BaseHookPoints):
    def get_hook_points(self, model):
        
        # text model
        self.hook_points["text_token_embeddings"] = ("text_model.embeddings.token_embedding.forward", model.text_model.embeddings.token_embedding.forward)
        self.hook_points["text_position_embeddings"] = ("text_model.embeddings.position_embedding.forward", model.text_model.embeddings.position_embedding.forward)
        for idx, layer in enumerate(model.text_model.encoder.layers):

            self.hook_points[f"text_model.{idx}.ln1"] = (f"text_model.encoder.layers[{idx}].layer_norm1.forward", layer.layer_norm1.forward)
            self.hook_points[f"text_model.{idx}.attn_q"] = (f"text_model.encoder.layers[{idx}].self_attn.q_proj.forward", layer.self_attn.q_proj.forward)
            self.hook_points[f"text_model.{idx}.attn_k"] = (f"text_model.encoder.layers[{idx}].self_attn.k_proj.forward", layer.self_attn.k_proj.forward)
            self.hook_points[f"text_model.{idx}.attn_v"] = (f"text_model.encoder.layers[{idx}].self_attn.v_proj.forward", layer.self_attn.v_proj.forward)
            self.hook_points[f"text_model.{idx}.attn_z"] = (f"text_model.encoder.layers[{idx}].self_attn.out_proj.forward", layer.self_attn.out_proj.forward)

            self.hook_points[f"text_model.{idx}.ln2"] = (f"text_model.encoder.layers[{idx}].layer_norm2.forward", layer.layer_norm2.forward)
            self.hook_points[f"text_model.{idx}.mlp.act_fn"] = (f"text_model.encoder.layers[{idx}].mlp.activation_fn.forward", layer.mlp.activation_fn.forward)
            self.hook_points[f"text_model.{idx}.mlp.fc1"] = (f"text_model.encoder.layers[{idx}].mlp.fc1.forward", layer.mlp.fc1.forward)
            self.hook_points[f"text_model.{idx}.mlp.fc2"] = (f"text_model.encoder.layers[{idx}].mlp.fc2.forward", layer.mlp.fc2.forward)
        self.hook_points[f"text_model.ln_final"] = (f"text_model.final_layer_norm.forward", model.text_model.final_layer_norm.forward)

        # vision model
        self.hook_points["vision_embeddings"] = ("vision_model.embeddings.forward", model.vision_model.embeddings.forward)
        self.hook_points["vision_embeddings"] = ("vision_model.embeddings.forward", model.vision_model.embeddings.forward)
        for idx, layer in enumerate(model.vision_model.encoder.layers):

            self.hook_points[f"vision_model.{idx}.ln1"] = (f"vision_model.encoder.layers[{idx}].layer_norm1.forward", layer.layer_norm1.forward)
            self.hook_points[f"vision_model.{idx}.attn_q"] = (f"vision_model.encoder.layers[{idx}].self_attn.q_proj.forward", layer.self_attn.q_proj.forward)
            self.hook_points[f"vision_model.{idx}.attn_k"] = (f"vision_model.encoder.layers[{idx}].self_attn.k_proj.forward", layer.self_attn.k_proj.forward)
            self.hook_points[f"vision_model.{idx}.attn_v"] = (f"vision_model.encoder.layers[{idx}].self_attn.v_proj.forward", layer.self_attn.v_proj.forward)
            self.hook_points[f"vision_model.{idx}.attn_z"] = (f"vision_model.encoder.layers[{idx}].self_attn.out_proj.forward", layer.self_attn.out_proj.forward)

            self.hook_points[f"vision_model.{idx}.ln2"] = (f"vision_model.encoder.layers[{idx}].layer_norm2.forward", layer.layer_norm2.forward)
            self.hook_points[f"vision_model.{idx}.mlp.act_fn"] = (f"vision_model.encoder.layers[{idx}].mlp.activation_fn.forward", layer.mlp.activation_fn.forward)
            self.hook_points[f"vision_model.{idx}.mlp.fc1"] = (f"vision_model.encoder.layers[{idx}].mlp.fc1.forward", layer.mlp.fc1.forward)
            self.hook_points[f"vision_model.{idx}.mlp.fc2"] = (f"vision_model.encoder.layers[{idx}].mlp.fc2.forward", layer.mlp.fc2.forward)
        
        self.hook_points["vision_model.ln1"] = ("vision_model.pre_layrnorm.forward", model.vision_model.pre_layrnorm.forward)
        self.hook_points["vision_model.ln_final"] = ("vision_model.post_layernorm.forward", model.vision_model.post_layernorm.forward)

        # joint-embedding space
        self.hook_points["visual_projection"] = ("visual_projectionforward", model.visual_projection.forward)
        self.hook_points["text_projection"] = ("text_projection.forward", model.text_projection.forward)        
        return self.hook_points


HOOK_POINT_MAPPING = OrderedDict([
    (CLIPConfig, CLIPHookPoints),
    (ViTConfig, ViTHookPoints),
])


def get_hook_points(model, config):

    if type(config) in HOOK_POINT_MAPPING.keys():
        cls = HOOK_POINT_MAPPING[type(config)]()
    else:
        raise EnvironmentError(
            f'No HookPoints specified for {config}.'
        )
    return cls.get_hook_points(model)
