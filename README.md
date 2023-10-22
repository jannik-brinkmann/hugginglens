# VisionTransformerLens

This is an extension of [TransformerLens](https://github.com/neelnanda-io/TransformerLens) that integrates other modalities using [HuggingFace](https://github.com/huggingface). It incorporates components of TransformerLens such as `HookPoints` and retains signature functionalities like `run_with_cache()`, to make it straightforward for users to explore multimodal mechanistic interpretability.  

### Installation
```setup
git clone git@github.com:jannik-brinkmann/vision-transformerlens.git
```

### Example
```usage
from PIL import Image
import requests
from transformers import CLIPProcessor

from lens import HookedVisionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load a model (e.g. CLIP)
model_name_or_path = "openai/clip-vit-base-patch32"
model = HookedVisionTransformer.from_pretrained(model_name_or_path, device)
processor = CLIPProcessor.from_pretrained(model_name_or_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# extract image features
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
inputs.to(device)

# run the model and get outputs and activations
outputs, activations = model.run_with_cache(**inputs)
```

### Citation
```
@misc{brinkmann2023visiontransformerlens
  title   = {VisionTransformerLens},
  author  = {Brinkmann, Jannik},
  journal = {https://github.com/jannik-brinkmann/VisionTransformerLens},
  year    = {2023}
}
```
