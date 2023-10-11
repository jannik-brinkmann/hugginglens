# HuggingLens

[![](https://img.shields.io/github/license/jannik-brinkmann/vision_transformer_lens.svg)](https://github.com/jannik-brinkmann/vision_transformer_lens/blob/master/LICENSE.md)

This is an extension of [TransformerLens](https://github.com/neelnanda-io/TransformerLens)

### Installation
To install, run:

```setup
pip install vision_transformer_lens
```

### 
```usage
import vision_transformer_lens
from transformers import AutoImageProcessor

# load a model (e.g. ViT Base)
model_name_or_path = "google/vit-base-patch16-224"
model = HookedVisionTransformer.from_pretrained(model_name_or_path)
image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)

# run the model and get logits and activations
features = image_processor(img, return_tensors="pt")
logits, activations = model.run_with_cache()
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