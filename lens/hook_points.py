import torch
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from transformer_lens.hook_points import HookedRootModule


NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str]]]


class HFHookedRootModule(HookedRootModule):

    def __init__(self, *args):
        super().__init__(*args)
    
    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device=None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> dict:
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif type(names_filter) == str:
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif type(names_filter) == list:
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        self.is_caching = True

        def save_hook(tensor, hook):

            # in HuggingFace's transformers, the forward() function might not return a tensor, but a dict with various outputs. Therefore, 
            # we check if the variable 'tensor' is a tensor, or if it's a dictionary that contains a tensor under the "hidden_states" key.
            if isinstance(tensor, torch.Tensor):
                pass
            elif isinstance(tensor, dict) and "hidden_states" in tensor and isinstance(tensor["hidden_states"], torch.Tensor):
                tensor = tensor["hidden_states"]
            else:
                return

            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor, hook):

            # see comment in 'save_hook'
            if isinstance(tensor, torch.Tensor):
                pass
            elif isinstance(tensor, dict) and "hidden_states" in tensor and isinstance(tensor["hidden_states"], torch.Tensor):
                tensor = tensor["hidden_states"]
            else:
                return

            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)

        fwd_hooks = []
        bwd_hooks = []
        for name, hp in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, save_hook))
                if incl_bwd:
                    bwd_hooks.append((name, save_hook_back))

        return cache, fwd_hooks, bwd_hooks
