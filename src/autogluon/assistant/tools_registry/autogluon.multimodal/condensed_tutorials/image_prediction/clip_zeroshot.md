# Condensed: ```python

Summary: This tutorial demonstrates implementing zero-shot image classification using AutoGluon's MultiModalPredictor with CLIP, enabling image classification without training data by matching images against text descriptions. It covers installing autogluon.multimodal, initializing a predictor with problem_type="zero_shot_image_classification", and using predict_proba() with image-text pairs to get classification probabilities. Key techniques include leveraging CLIP's contrastive learning on 400M image-text pairs for arbitrary visual classification tasks, handling uncommon object classes, and rapid prototyping. The tutorial highlights critical limitations like vulnerability to typographic attacks where overlaid text can manipulate predictions, and provides best practices for effective zero-shot classification scenarios without fine-tuning requirements.

*This is a condensed version that preserves essential implementation details and context.*

# Zero-Shot Image Classification with CLIP

## Installation & Setup

```python
!pip install autogluon.multimodal
```

## Basic Usage

**Key Concept**: CLIP (Contrastive Language-Image Pre-training) performs zero-shot classification without training data—just provide potential visual categories as text.

```python
from autogluon.multimodal import MultiModalPredictor

# Initialize predictor
predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")

# Predict with image and text candidates
prob = predictor.predict_proba(
    {"image": [dog_image]}, 
    {"text": ['This is a Husky', 'This is a Golden Retriever', 
              'This is a German Sheperd', 'This is a Samoyed.']}
)
```

## How CLIP Works

- **Training**: Pre-trained on 400M image-text pairs using contrastive learning
- **Objective**: Predicts which text snippet is paired with a given image
- **Advantage**: Applies to arbitrary visual classification tasks without fine-tuning

## Critical Limitations

⚠️ **Typographic Attacks**: CLIP is vulnerable to text overlaid on images

**Example**: An apple image correctly classified as "Granny Smith" (99%+ confidence) is misclassified as "iPod" when "iPod" text is added to the image.

```python
# Same classification code, different results based on text in image
prob = predictor.predict_proba(
    {"image": [image_with_text]}, 
    {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']}
)
```

## Best Practices

- Works well for uncommon object classes not in standard datasets
- Effective for quick prototyping without training data
- See [CLIP paper](https://arxiv.org/abs/2103.00020) for additional limitations
- For customization: refer to [Customize AutoMM](../advanced_topics/customization.ipynb)