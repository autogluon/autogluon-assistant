# Condensed: ```python

Summary: This tutorial demonstrates zero-shot image classification using CLIP via AutoGluon's MultiModalPredictor. It covers implementation of image classification without prior training by comparing images against arbitrary text descriptions. Key functionalities include initializing the predictor, performing predictions with image-text pairs, and understanding CLIP's capabilities and limitations (particularly typographic attacks). The code shows how to classify dog breeds and uncommon objects, making it useful for developers implementing flexible image classification systems without labeled training data for specific categories.

*This is a condensed version that preserves essential implementation details and context.*

# Zero-Shot Image Classification with CLIP

## Setup and Basic Usage

```python
!pip install autogluon.multimodal

from IPython.display import Image, display
from autogluon.multimodal.utils import download
from autogluon.multimodal import MultiModalPredictor

# Load a dog image
url = "https://farm4.staticflickr.com/3445/3262471985_ed886bf61a_z.jpg"
dog_image = download(url)
display(Image(filename=dog_image))
```

## Zero-Shot Classification Implementation

CLIP allows for zero-shot classification without training on specific datasets:

```python
# Initialize the predictor for zero-shot classification
predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")

# Predict dog breed
prob = predictor.predict_proba(
    {"image": [dog_image]}, 
    {"text": ['This is a Husky', 'This is a Golden Retriever', 
              'This is a German Sheperd', 'This is a Samoyed.']}
)
print("Label probs:", prob)
```

## Additional Example - Uncommon Object

```python
# Segway example
url = "https://live.staticflickr.com/7236/7114602897_9cf00b2820_b.jpg"
segway_image = download(url)
display(Image(filename=segway_image))

prob = predictor.predict_proba(
    {"image": [segway_image]}, 
    {"text": ['segway', 'bicycle', 'wheel', 'car']}
)
print("Label probs:", prob)
```

## How CLIP Works

CLIP (Contrastive Language-Image Pre-training) was trained on 400M image-text pairs using a contrastive learning approach. It predicts which text is paired with a given image, enabling application to arbitrary visual classification tasks.

## CLIP Limitations - Typographic Attacks

CLIP is vulnerable to text in images affecting predictions:

```python
# Apple example
url = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
image_path = download(url)
display(Image(filename=image_path))

# Normal classification
prob = predictor.predict_proba(
    {"image": [image_path]}, 
    {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']}
)
print("Label probs:", prob)

# Apple with "iPod" text
url = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
image_path = download(url)
display(Image(filename=image_path))

# Classification affected by text
prob = predictor.predict_proba(
    {"image": [image_path]}, 
    {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']}
)
print("Label probs:", prob)
```

For more details on CLIP's limitations, refer to the [CLIP paper](https://arxiv.org/abs/2103.00020).

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)