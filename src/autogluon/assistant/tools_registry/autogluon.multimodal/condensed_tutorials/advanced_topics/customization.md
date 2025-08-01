# Condensed: ```

Summary: This tutorial covers AutoMM optimization parameters for fine-tuning multimodal models. It demonstrates how to configure learning rates, optimizers, training schedules, and validation strategies through hyperparameter settings. Key features include layer-wise learning rate decay, gradient handling, parameter-efficient fine-tuning (PEFT), GPU/batch size configuration, precision settings, and model-specific options for text (tokenization, augmentation), image (transforms, backbones), and object detection models. The tutorial also covers data processing strategies for handling missing values, label preprocessing, and advanced techniques like Mixup augmentation and knowledge distillation. These configurations help optimize model performance for various multimodal machine learning tasks.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Optimization Parameters

## Learning Rate and Optimizer Settings

```python
# Set learning rate
predictor.fit(hyperparameters={"optim.lr": 5.0e-4})  # Default: 1.0e-4

# Choose optimizer type
predictor.fit(hyperparameters={"optim.optim_type": "adam"})  # Default: "adamw"
# Options: "sgd", "adam", "adamw"

# Set weight decay
predictor.fit(hyperparameters={"optim.weight_decay": 1.0e-4})  # Default: 1.0e-3
```

## Learning Rate Strategies

```python
# Layer-wise learning rate decay
predictor.fit(hyperparameters={"optim.lr_decay": 0.9})  # Default
# Set to 1 for uniform learning rate

# Two-stage learning rate
predictor.fit(hyperparameters={"optim.lr_mult": 10})  # Default: 1
# Head layer gets lr * lr_mult, other layers get lr

# Learning rate strategy selection
predictor.fit(hyperparameters={"optim.lr_choice": "two_stages"})  # Default: "layerwise_decay"
```

## Training Schedule

```python
# Learning rate schedule
predictor.fit(hyperparameters={"optim.lr_schedule": "polynomial_decay"})  # Default: "cosine_decay"
# Options: "cosine_decay", "polynomial_decay", "linear_decay"

# Training duration
predictor.fit(hyperparameters={"optim.max_epochs": 20})  # Default: 10
predictor.fit(hyperparameters={"optim.max_steps": 100})  # Default: -1 (disabled)

# Learning rate warmup
predictor.fit(hyperparameters={"optim.warmup_steps": 0.2})  # Default: 0.1
# Percentage of steps to warm up from 0 to full lr
```

## Validation and Early Stopping

```python
# Early stopping patience
predictor.fit(hyperparameters={"optim.patience": 5})  # Default: 10

# Validation check frequency
predictor.fit(hyperparameters={"optim.val_check_interval": 0.25})  # Default: 0.5
# Float (0-1): fraction of epoch, Int: number of batches
```

## Gradient Handling

```python
# Gradient clipping
predictor.fit(hyperparameters={"optim.gradient_clip_algorithm": "value"})  # Default: "norm"
predictor.fit(hyperparameters={"optim.gradient_clip_val": 5})  # Default: 1

# Gradient norm tracking
predictor.fit(hyperparameters={"optim.track_grad_norm": 2})  # Default: -1 (no tracking)
```

## Logging and Model Selection

```python
# Logging frequency
predictor.fit(hyperparameters={"optim.log_every_n_steps": 50})  # Default: 10

# Model checkpoint averaging
predictor.fit(hyperparameters={"optim.top_k": 5})  # Default: 3
predictor.fit(hyperparameters={"optim.top_k_average_method": "uniform_soup"})  # Default: "greedy_soup"
# Options: "greedy_soup", "uniform_soup", "best"
```

## Parameter-Efficient Fine-Tuning (PEFT)

```python
# PEFT options
predictor.fit(hyperparameters={"optim.peft": "bit_fit"})  # Default: None
# Options: "bit_fit", "norm_fit", "lora", "lora_bias", "lora_norm", "ia3", "ia3_bias", "ia3_norm"
```

# AutoMM Hyperparameter Configuration Guide (Part 2/5)

## Optimization Parameters

```python
# Skip final validation
predictor.fit(hyperparameters={"optim.skip_final_val": True})
```

## Environment Configuration

### GPU and Batch Size Settings
```python
# GPU configuration
predictor.fit(hyperparameters={"env.num_gpus": -1})  # Use all available GPUs
predictor.fit(hyperparameters={"env.num_gpus": 1})   # Use 1 GPU only

# Batch size settings
predictor.fit(hyperparameters={"env.per_gpu_batch_size": 16})  # Batch size per GPU
predictor.fit(hyperparameters={"env.batch_size": 256})         # Total effective batch size
predictor.fit(hyperparameters={"env.inference_batch_size_ratio": 2})  # 2x batch size during inference
```

### Precision and Workers
```python
# Precision settings
predictor.fit(hyperparameters={"env.precision": "16-mixed"})    # Default mixed precision
predictor.fit(hyperparameters={"env.precision": "bf16-mixed"})  # bfloat16 mixed precision

# Worker processes
predictor.fit(hyperparameters={"env.num_workers": 4})  # Training dataloader workers
predictor.fit(hyperparameters={"env.num_workers_inference": 4})  # Inference dataloader workers
```

### Training Strategy and Hardware
```python
# Distributed training mode
predictor.fit(hyperparameters={"env.strategy": "ddp"})  # Distributed data parallel

# Hardware accelerator
predictor.fit(hyperparameters={"env.accelerator": "cpu"})  # Force CPU training
```

### PyTorch Compilation
```python
# Enable torch.compile
predictor.fit(hyperparameters={
    "env.compile.turn_on": True,
    "env.compile.mode": "reduce-overhead",  # Good for small batches
    "env.compile.dynamic": False,           # Static input shapes
    "env.compile.backend": "inductor"       # Default backend
})
```

## Model Configuration

### Model Selection
```python
# Choose specific model types
predictor.fit(hyperparameters={"model.names": ["hf_text"]})  # Text models only
predictor.fit(hyperparameters={"model.names": ["timm_image"]})  # Image models only
predictor.fit(hyperparameters={"model.names": ["clip"]})  # CLIP models only
```

### Text Model Configuration
```python
# Hugging Face text model settings
predictor.fit(hyperparameters={
    "model.hf_text.checkpoint_name": "roberta-base",  # Choose specific text backbone
    "model.hf_text.pooling_mode": "mean"  # Use mean pooling instead of CLS token
})
```

# AutoMM Text and Image Model Configuration

## Text Model Configuration

### Tokenizer Selection
```python
# Default auto tokenizer
predictor.fit(hyperparameters={"model.hf_text.tokenizer_name": "hf_auto"})

# Using ELECTRA tokenizer
predictor.fit(hyperparameters={"model.hf_text.tokenizer_name": "electra"})
```
Options include: `hf_auto`, `bert`, `electra`, and `clip`

### Text Length Configuration
```python
# Default maximum length
predictor.fit(hyperparameters={"model.hf_text.max_text_len": 512})

# Use tokenizer's maximum allowed length
predictor.fit(hyperparameters={"model.hf_text.max_text_len": -1})
```

### Text Processing Options
```python
# Insert SEP token between texts from different columns (default)
predictor.fit(hyperparameters={"model.hf_text.insert_sep": True})

# Number of text segments in token sequence (default: 2)
predictor.fit(hyperparameters={"model.hf_text.text_segment_num": 2})

# Handle long text sequences (default: cut from beginning)
predictor.fit(hyperparameters={"model.hf_text.stochastic_chunk": False})
```

### Text Augmentation
```python
# Minimum token length for text augmentation (default: 10)
predictor.fit(hyperparameters={"model.hf_text.text_aug_detect_length": 10})

# Maximum percentage for text augmentation (default: 0 - disabled)
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0})
# Enable with 10% maximum scale
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0.1})
```

### Memory Optimization
```python
# Enable gradient checkpointing to reduce memory usage
predictor.fit(hyperparameters={"model.hf_text.gradient_checkpointing": True})
```

## FT-Transformer Configuration

### Model Architecture
```python
# Initialize from checkpoint
predictor.fit(hyperparameters={"model.ft_transformer.checkpoint_name": "my_checkpoint.ckpt"})
# Or from URL
predictor.fit(hyperparameters={"model.ft_transformer.checkpoint_name": "https://automl-mm-bench.s3.amazonaws.com/ft_transformer_pretrained_ckpt/iter_2k.ckpt"})

# Number of transformer blocks (default: 3)
predictor.fit(hyperparameters={"model.ft_transformer.num_blocks": 5})

# Token dimension (default: 192)
predictor.fit(hyperparameters={"model.ft_transformer.token_dim": 256})

# Model embedding dimension (default: 192)
predictor.fit(hyperparameters={"model.ft_transformer.hidden_size": 256})

# FFN hidden layer dimension (default: 192)
predictor.fit(hyperparameters={"model.ft_transformer.ffn_hidden_size": 256})
```

## Image Model Configuration

### Model Selection
```python
# Default Swin Transformer
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "swin_base_patch4_window7_224"})

# Use ViT base
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "vit_base_patch32_224"})
```

### Image Augmentation
```python
# Default transforms
predictor.fit(hyperparameters={"model.timm_image.train_transforms": [
    "resize_shorter_side", "center_crop", "trivial_augment"
]})

# Custom transforms
predictor.fit(hyperparameters={"model.timm_image.train_transforms": [
    "random_resize_crop", "random_horizontal_flip"
]})

# Using torchvision transforms
predictor.fit(hyperparameters={"model.timm_image.train_transforms": [
    torchvision.transforms.RandomResizedCrop(224), 
    torchvision.transforms.RandomHorizontalFlip()
]})
```

# AutoMM Tutorial: Configuration Options (Chunk 4/5)

## Image Transformation Options

### model.timm_image.val_transforms
Transform images for validation/test/deployment:
```python
# Default transforms
predictor.fit(hyperparameters={"model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"]})

# Square resize
predictor.fit(hyperparameters={"model.timm_image.val_transforms": ["resize_to_square"]})

# Custom transforms
predictor.fit(hyperparameters={"model.timm_image.val_transforms": [torchvision.transforms.Resize((224, 224)]})
```

## Object Detection Models

### model.mmdet_image.checkpoint_name
Specify a MMDetection model:
```python
# Default model
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_8xb24-320-300e_coco"})

# YOLOX-L
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "yolox_l"})

# DINO-SwinL
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "dino-5scale_swin-l_8xb2-36e_coco"})
```

### model.mmdet_image.output_bbox_format
Bounding box format:
- `"xyxy"`: [x1,y1,x2,y2] (default)
- `"xywh"`: [x1,y1,w,h]

### model.mmdet_image.frozen_layers
Freeze specific layers:
```python
# Default - freeze nothing
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.frozen_layers": []})

# Freeze backbone
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.frozen_layers": ["backbone"]})
```

## Segment Anything Model (SAM) Configuration


...(truncated)