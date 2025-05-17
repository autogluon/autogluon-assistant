# Condensed: Customize AutoMM

Summary: This tutorial provides comprehensive implementation guidance for AutoMM (Auto Multi-Modal) model configurations, covering optimization techniques, model architectures, and data processing. It details how to configure learning rates, optimizers, gradient management, and model checkpointing; implement efficient fine-tuning strategies like LoRA and IA3; set up various model architectures including HF-Text, TIMM-Image, CLIP, and SAM; and handle data preprocessing, augmentation (Mixup/CutMix), and distillation. Key functionalities include GPU/batch size configuration, precision settings, model compilation options, text/image transformations, and specialized configurations for classification, object detection, and segmentation tasks. The tutorial is particularly valuable for tasks involving model training optimization, multi-modal learning, and performance tuning.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and configurations:

# AutoMM Customization Guide

## Key Optimization Configurations

### Learning Rate Related
```python
# Basic learning rate
predictor.fit(hyperparameters={"optimization.learning_rate": 1.0e-4})

# Learning rate decay (for layer-wise adjustments)
predictor.fit(hyperparameters={"optimization.lr_decay": 0.9})  # 1.0 to disable

# Two-stage learning rate
predictor.fit(hyperparameters={"optimization.lr_mult": 10})    # Multiplier for head layer
predictor.fit(hyperparameters={"optimization.lr_choice": "two_stages"})  # or "layerwise_decay"
```

### Optimizer Settings
```python
# Optimizer selection
predictor.fit(hyperparameters={"optimization.optim_type": "adamw"})  # Options: sgd, adam, adamw

# Weight decay
predictor.fit(hyperparameters={"optimization.weight_decay": 1.0e-3})

# Learning rate schedule
predictor.fit(hyperparameters={"optimization.lr_schedule": "cosine_decay"})  # Options: polynomial_decay, linear_decay
```

### Training Control
```python
# Training duration
predictor.fit(hyperparameters={
    "optimization.max_epochs": 10,
    "optimization.max_steps": -1,  # -1 to disable
    "optimization.warmup_steps": 0.1  # Percentage of steps for warmup
})

# Early stopping
predictor.fit(hyperparameters={
    "optimization.patience": 10,
    "optimization.val_check_interval": 0.5  # Check validation every 50% of epoch
})
```

### Gradient Management
```python
# Gradient clipping
predictor.fit(hyperparameters={
    "optimization.gradient_clip_algorithm": "norm",  # or "value"
    "optimization.gradient_clip_val": 1
})

# Gradient tracking
predictor.fit(hyperparameters={"optimization.track_grad_norm": 2})  # -1 to disable
```

### Model Checkpointing
```python
# Model averaging settings
predictor.fit(hyperparameters={
    "optimization.top_k": 3,  # Number of checkpoints to consider
    "optimization.top_k_average_method": "greedy_soup"  # Options: uniform_soup, best
})
```

## Important Notes:
- Use `lr_decay=1` for uniform learning rate across layers
- Set `max_steps=-1` to control training by epochs only
- `val_check_interval` accepts float (0-1) or int values
- `greedy_soup` averaging stops if performance decreases
- Gradient tracking may impact performance, disable if not needed

Here's the condensed tutorial content focusing on key implementation details and practices:

# AutoMM Hyperparameter Configuration Guide - Part 2

## Optimization Parameters

### optimization.top_k_average_method
```python
# Average top k checkpoints uniformly
predictor.fit(hyperparameters={"optimization.top_k_average_method": "uniform_soup"})
```

### optimization.efficient_finetune
Parameter-efficient finetuning options:
- `bit_fit`: Bias parameters only
- `norm_fit`: Normalization + bias parameters  
- `lora`: LoRA Adaptors
- `lora_bias`: LoRA + bias
- `lora_norm`: LoRA + normalization + bias
- `ia3`: IA3 algorithm
- `ia3_bias`: IA3 + bias
- `ia3_norm`: IA3 + normalization + bias

```python
# Example configurations
predictor.fit(hyperparameters={"optimization.efficient_finetune": "bit_fit"})
predictor.fit(hyperparameters={"optimization.efficient_finetune": "ia3_bias"})
```

## Environment Settings

### GPU and Batch Size Configuration
```python
# GPU settings
predictor.fit(hyperparameters={
    "env.num_gpus": -1,  # Use all available GPUs
    "env.per_gpu_batch_size": 8,
    "env.batch_size": 128,
    "env.eval_batch_size_ratio": 4
})
```

### Precision and Workers
```python
# Training precision
predictor.fit(hyperparameters={
    "env.precision": "16-mixed",  # Supports: 64, 32, bf16-mixed, 16-mixed
    "env.num_workers": 2,
    "env.num_workers_evaluation": 2
})
```

### Distribution Strategy
```python
# Training strategy
predictor.fit(hyperparameters={
    "env.strategy": "ddp_spawn",  # Options: dp, ddp, ddp_spawn
    "env.accelerator": "auto"     # Options: cpu, gpu, auto
})
```

### Model Compilation
```python
# Torch compile settings
predictor.fit(hyperparameters={
    "env.compile.turn_on": False,
    "env.compile.mode": "default",
    "env.compile.dynamic": True,
    "env.compile.backend": "inductor"
})
```

## Important Notes:
- Mixed precision (16-mixed) can achieve 3x speedups on modern GPUs
- More workers don't always improve speed, especially with ddp_spawn
- Model compilation is recommended for large models and long training sessions
- Batch size accumulation occurs if env.batch_size > (per_gpu_batch_size * num_gpus)

This condensed version maintains the critical implementation details while removing redundant examples and explanations.

Here's the condensed tutorial content focusing on key implementation details and configurations:

# AutoMM Model Configuration Guide

## Core Model Selection
```python
# Select specific model types
predictor.fit(hyperparameters={
    "model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"] # Default
    # OR
    "model.names": ["hf_text"]  # Text only
    "model.names": ["timm_image"]  # Image only 
    "model.names": ["clip"]  # CLIP only
})
```

## Text Model Configurations (HF_Text)

### Key Parameters
```python
predictor.fit(hyperparameters={
    # Model checkpoint
    "model.hf_text.checkpoint_name": "google/electra-base-discriminator",  # Default
    
    # Pooling configuration
    "model.hf_text.pooling_mode": "cls",  # Options: "cls" or "mean"
    
    # Tokenizer selection
    "model.hf_text.tokenizer_name": "hf_auto",  # Options: "hf_auto", "bert", "electra", "clip"
    
    # Text processing
    "model.hf_text.max_text_len": 512,  # Use -1 for model's max length
    "model.hf_text.insert_sep": True,  # Insert SEP token between text columns
    "model.hf_text.text_segment_num": 2,  # Number of text segments per sequence
    
    # Text augmentation
    "model.hf_text.stochastic_chunk": False,  # Random text chunk selection
    "model.hf_text.text_aug_detect_length": 10,  # Min length for augmentation
    "model.hf_text.text_trivial_aug_maxscale": 0,  # Max % of tokens for augmentation
    
    # Performance optimization
    "model.hf_text.gradient_checkpointing": False  # Memory optimization
})
```

## FT Transformer Configurations

### Key Parameters
```python
predictor.fit(hyperparameters={
    # Model initialization
    "model.ft_transformer.checkpoint_name": None,  # Load from local or URL
    
    # Architecture
    "model.ft_transformer.num_blocks": 3,
    "model.ft_transformer.token_dim": 192,
    "model.ft_transformer.hidden_size": 192,
    "model.ft_transformer.ffn_hidden_size": 192
})
```

## Important Notes:
- Text augmentation only occurs when text length ≥ `text_aug_detect_length`
- Gradient checkpointing reduces memory usage but may impact training speed
- For `max_text_len`, system uses minimum between specified value and model's maximum
- Token segments are limited by model's default maximum

This condensed version maintains all critical implementation details while removing redundant examples and verbose explanations.

Here's the condensed tutorial focusing on key implementation details and configurations:

# Image Model Configurations

### Checkpoint Selection
```python
# Swin Transformer (default)
predictor.fit(hyperparameters={
    "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224"
})

# ViT Base
predictor.fit(hyperparameters={
    "model.timm_image.checkpoint_name": "vit_base_patch32_224"
})
```

### Image Transforms
```python
# Training transforms
predictor.fit(hyperparameters={
    "model.timm_image.train_transforms": [
        "resize_shorter_side", 
        "center_crop", 
        "trivial_augment"  # default
    ]
})

# Custom transforms using torchvision
predictor.fit(hyperparameters={
    "model.timm_image.train_transforms": [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip()
    ]
})

# Validation transforms
predictor.fit(hyperparameters={
    "model.timm_image.val_transforms": [
        "resize_shorter_side",
        "center_crop"  # default
    ]
})
```

# Object Detection Configurations

### MMDetection Models
```python
# Default YOLOv3
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_8xb24-320-300e_coco"
})

# YOLOX-L
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.checkpoint_name": "yolox_l"
})
```

### Important Settings
```python
# Bounding box format
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.output_bbox_format": "xyxy"  # or "xywh"
})

# Freeze layers
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.frozen_layers": ["backbone", "neck"]
})
```

# SAM (Segment Anything Model) Configurations

```python
# Model selection
predictor.fit(hyperparameters={
    "model.sam.checkpoint_name": "facebook/sam-vit-huge"  # default
})

# Training configuration
predictor.fit(hyperparameters={
    "model.sam.train_transforms": ["random_horizontal_flip"],
    "model.sam.img_transforms": ["resize_to_square"],
    "model.sam.gt_transforms": ["resize_gt_to_square"],
    "model.sam.num_mask_tokens": 1,
    "model.sam.ignore_label": 255
})
```

# Data Processing Configurations

### Missing Data Handling
```python
predictor.fit(hyperparameters={
    "data.image.missing_value_strategy": "zero"  # or "skip"
})
```

### Data Type Conversions
```python
predictor.fit(hyperparameters={
    "data.categorical.convert_to_text": False,
    "data.numerical.convert_to_text": False,
    "data.text.normalize_text": False
})
```


...(truncated)