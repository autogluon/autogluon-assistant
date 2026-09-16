# Condensed: ```

Summary: This tutorial provides comprehensive hyperparameter configuration for AutoGluon's MultiModalPredictor, covering optimization (learning rates with layer-wise decay, two-stage strategies, schedulers, optimizers like AdamW, early stopping, gradient clipping, checkpoint averaging with greedy/uniform soup, and PEFT methods like LoRA/IA3), environment setup (GPU allocation, batch sizing with gradient accumulation, mixed precision training, torch.compile optimization, distributed training strategies like DDP, and dataloader workers), multi-modal model configuration (text models from Huggingface with tokenization and augmentation, TIMM/CLIP image models with transforms, FT-Transformer for tabular data, MMDetection for object detection, SAM for segmentation), data processing (missing value handling, text normalization, categorical/numerical conversion and scaling, label preprocessing, Mixup/Cutmix augmentation), and knowledge distillation parameters. Essential for tasks involving multi-modal ML training, fine-tuning pretrained models, implementing parameter-efficient training, configuring distributed training, and applying advanced augmentation techniques.

*This is a condensed version that preserves essential implementation details and context.*

# Optimization Hyperparameters

## Learning Rate Configuration

### optim.lr
Learning rate for optimization.
```python
predictor.fit(hyperparameters={"optim.lr": 1.0e-4})  # default
predictor.fit(hyperparameters={"optim.lr": 5.0e-4})  # custom
```

### optim.lr_decay
Layer-wise learning rate decay. Layer `i` in `n`-layer model gets: `optim.lr * optim.lr_decay^(n-i)`. Set to `1` for uniform learning rate.
```python
predictor.fit(hyperparameters={"optim.lr_decay": 0.9})  # default
predictor.fit(hyperparameters={"optim.lr_decay": 1})    # uniform lr
```

### optim.lr_mult
Two-stage learning rate multiplier. Head layer gets `optim.lr * optim.lr_mult`, other layers get `optim.lr`.
```python
predictor.fit(hyperparameters={"optim.lr_mult": 1})   # default
predictor.fit(hyperparameters={"optim.lr_mult": 10})  # 10x head lr
```

### optim.lr_choice
Learning rate strategy: `"two_stages"` (uses `lr_mult`), `"layerwise_decay"` (uses `lr_decay`), or `""` (uniform).
```python
predictor.fit(hyperparameters={"optim.lr_choice": "layerwise_decay"})  # default
predictor.fit(hyperparameters={"optim.lr_choice": "two_stages"})
```

### optim.lr_schedule
Learning rate schedule: `"cosine_decay"` (default), `"polynomial_decay"`, or `"linear_decay"`.
```python
predictor.fit(hyperparameters={"optim.lr_schedule": "cosine_decay"})
```

### optim.warmup_steps
Warmup learning rate from 0 to `optim.lr` over this fraction of steps.
```python
predictor.fit(hyperparameters={"optim.warmup_steps": 0.1})  # default: 10% of steps
```

## Optimizer Configuration

### optim.optim_type
Optimizer: `"sgd"`, `"adam"`, or `"adamw"` (default, decouples weight decay).
```python
predictor.fit(hyperparameters={"optim.optim_type": "adamw"})  # default
```

### optim.weight_decay
Weight decay regularization.
```python
predictor.fit(hyperparameters={"optim.weight_decay": 1.0e-3})  # default
```

## Training Duration

### optim.max_epochs
Maximum training epochs.
```python
predictor.fit(hyperparameters={"optim.max_epochs": 10})  # default
```

### optim.max_steps
Maximum training steps. Training stops at earliest of `max_steps` or `max_epochs`. Set to `-1` to disable.
```python
predictor.fit(hyperparameters={"optim.max_steps": -1})   # default: disabled
predictor.fit(hyperparameters={"optim.max_steps": 100})  # stop at 100 steps
```

## Validation & Early Stopping

### optim.val_check_interval
Validation frequency per epoch. Float `[0.0, 1.0]` for fraction of epoch, int for number of batches.
```python
predictor.fit(hyperparameters={"optim.val_check_interval": 0.5})   # default: 2x per epoch
predictor.fit(hyperparameters={"optim.val_check_interval": 0.25})  # 4x per epoch
```

### optim.patience
Early stopping: stop after this many checks without improvement.
```python
predictor.fit(hyperparameters={"optim.patience": 10})  # default
```

### optim.skip_final_val
Skip final validation after training stops.

## Gradient Management

### optim.gradient_clip_algorithm
Gradient clipping method: `"norm"` (default) or `"value"`.
```python
predictor.fit(hyperparameters={"optim.gradient_clip_algorithm": "norm"})
```

### optim.gradient_clip_val
Gradient clipping threshold (absolute value or norm based on algorithm).
```python
predictor.fit(hyperparameters={"optim.gradient_clip_val": 1})  # default
```

### optim.track_grad_norm
Track p-norm of gradients. Set to `-1` to disable, `2` for 2-norm, or `'inf'` for infinity-norm.
```python
predictor.fit(hyperparameters={"optim.track_grad_norm": -1})  # default: disabled
```

## Model Checkpointing

### optim.top_k
Number of best checkpoints to use for model averaging.
```python
predictor.fit(hyperparameters={"optim.top_k": 3})  # default
```

### optim.top_k_average_method
Checkpoint averaging strategy:
- `"greedy_soup"` (default): adds checkpoints best-to-worst, stops if performance decreases
- `"uniform_soup"`: averages all top k checkpoints
- `"best"`: uses single best checkpoint

```python
predictor.fit(hyperparameters={"optim.top_k_average_method": "greedy_soup"})
```

## Parameter-Efficient Fine-Tuning (PEFT)

### optim.peft
Fine-tune only a subset of parameters:
- `"bit_fit"`: bias only
- `"norm_fit"`: normalization + bias
- `"lora"`: LoRA adaptors
- `"lora_bias"`: LoRA + bias
- `"lora_norm"`: LoRA + normalization + bias
- `"ia3"`: IA3 algorithm
- `"ia3_bias"`: IA3 + bias
- `"ia3_norm"`: IA3 + normalization + bias

```python
predictor.fit(hyperparameters={"optim.peft": None})        # default: full fine-tuning
predictor.fit(hyperparameters={"optim.peft": "bit_fit"})   # bias only
predictor.fit(hyperparameters={"optim.peft": "ia3_bias"})  # IA3 + bias
```

## Logging

### optim.log_every_n_steps
Logging frequency in steps.
```python
predictor.fit(hyperparameters={"optim.log_every_n_steps": 10})  # default
```

# Environment Configuration

## GPU & Batch Settings

### env.num_gpus
Number of GPUs to use. Set to `-1` to auto-detect all available GPUs.
```python
predictor.fit(hyperparameters={"env.num_gpus": -1})  # default: all GPUs
predictor.fit(hyperparameters={"env.num_gpus": 1})   # single GPU
```

### env.per_gpu_batch_size
Batch size per GPU.
```python
predictor.fit(hyperparameters={"env.per_gpu_batch_size": 8})  # default
```

### env.batch_size
Effective batch size per optimization step. If larger than `per_gpu_batch_size * num_gpus`, gradients accumulate over multiple steps.
```python
predictor.fit(hyperparameters={"env.batch_size": 128})  # default
```
**Accumulation steps** = `batch_size // (per_gpu_batch_size * num_gpus)`

### env.inference_batch_size_ratio
Multiplier for per-GPU batch size during prediction/evaluation: `per_gpu_batch_size * inference_batch_size_ratio`.
```python
predictor.fit(hyperparameters={"env.inference_batch_size_ratio": 4})  # default: 4x larger
```

## Precision & Performance

### env.precision
Training precision. Mixed precision (`"16-mixed"`, `"bf16-mixed"`) reduces memory and can achieve 3x+ speedup on modern GPUs.
- Double: `64`, `"64"`, `"64-true"`
- Float: `32`, `"32"`, `"32-true"`
- BFloat16: `"bf16-mixed"`, `"bf16-true"`
- Float16: `"16-mixed"`, `"16-true"`

```python
predictor.fit(hyperparameters={"env.precision": "16-mixed"})    # default
predictor.fit(hyperparameters={"env.precision": "bf16-mixed"})  # bfloat16
```

### env.compile.turn_on
Enable PyTorch model compilation via `torch.compile`. **Recommended for large models and long training**. Compilation adds upfront time cost.
```python
predictor.fit(hyperparameters={"env.compile.turn_on": False})  # default
predictor.fit(hyperparameters={"env.compile.turn_on": True})
```

### env.compile.mode
Compilation mode:
- `"default"`: standard compilation
- `"reduce-overhead"`: reduces Python overhead with CUDA graphs (useful for small batches)
- `"max-autotune"`: aggressive optimization
- `"max-autotune-no-cudagraphs"`: optimization without CUDA graphs

```python
predictor.fit(hyperparameters={"env.compile.mode": "default"})          # default
predictor.fit(hyperparameters={"env.compile.mode": "reduce-overhead"})
```

### env.compile.dynamic
Dynamic shape tracing. Set to `False` for static input shapes across batches.
```python
predictor.fit(hyperparameters={"env.compile.dynamic": True})  # default
```

### env.compile.backend
Compilation backend.
```python
predictor.fit(hyperparameters={"env.compile.backend": "inductor"})  # default
```

## Distributed Training

### env.strategy
Distributed training mode:
- `"dp"`: data parallel
- `"ddp"`: distributed data parallel (script-based)
- `"ddp_spawn"`: distributed data parallel (spawn-based, default)

```python
predictor.fit(hyperparameters={"env.strategy": "ddp_spawn"})  # default
predictor.fit(hyperparameters={"env.strategy": "ddp"})
```

### env.accelerator
Hardware accelerator: `"cpu"`, `"gpu"`, or `"auto"` (default, prioritizes GPU if available).
```python
predictor.fit(hyperparameters={"env.accelerator": "auto"})  # default
predictor.fit(hyperparameters={"env.accelerator": "cpu"})
```

## DataLoader Workers

### env.num_workers
Worker processes for training dataloader. **Note**: More workers don't always improve speed, especially with `strategy="ddp_spawn"`.
```python
predictor.fit(hyperparameters={"env.num_workers": 2})  # default
```

### env.num_workers_inference
Worker processes for prediction/evaluation dataloader.
```python
predictor.fit(hyperparameters={"env.num_workers_inference": 2})  # default
```

# Model Configuration

## Model Selection

### model.names
Model types to use. Automatically removes types for undetected modalities.
- `"hf_text"`: Huggingface text models
- `"timm_image"`: TIMM image models
- `"clip"`: CLIP models
- `"categorical_mlp"`: MLP for categorical data
- `"numerical_mlp"`: MLP for numerical data
- `"ft_transformer"`: FT-Transformer for tabular data
- `"fusion_mlp"`: MLP-based multi-backbone fusion
- `"fusion_transformer"`: Transformer-based multi-backbone fusion
- `"sam"`: Segment Anything Model

```python
# default
predictor.fit(hyperparameters={"model.names": ["hf_text", "timm_image", "clip", 
                                               "categorical_mlp", "numerical_mlp", "fusion_mlp"]})

# text only
predictor.fit(hyperparameters={"model.names": ["hf_text"]})

# image only
predictor.fit(hyperparameters={"model.names": ["timm_image"]})

# CLIP only
predictor.fit(hyperparameters={"model.names": ["clip"]})
```

## Text Model Configuration

### model.hf_text.checkpoint_name
Huggingface text backbone (any AutoModel-supported model).
```python
predictor.fit(hyperparameters={"model.hf_text.checkpoint_name": "google/electra-base-discriminator"})  # default
predictor.fit(hyperparameters={"model.hf_text.checkpoint_name": "roberta-base"})
```

### model.hf_text.pooling_mode
Feature pooling for transformer architectures:
- `"cls"`: use CLS token (default)
- `"mean"`: average all token features

```python
predictor.fit(hyperparameters={"model.hf_text.pooling_mode": "cls"})   # default
predictor.fit(hyperparameters={"model.hf_text.pooling_mode": "mean"})
```

# Text Model Configuration (Continued)

## Tokenization

### model.hf_text.tokenizer_name
Text tokenizer selection. **Recommended: use default auto tokenizer.**
- `"hf_auto"`: Huggingface auto tokenizer (default)
- `"bert"`: BERT tokenizer
- `"electra"`: ELECTRA tokenizer
- `"clip"`: CLIP tokenizer

```python
predictor.fit(hyperparameters={"model.hf_text.tokenizer_name": "hf_auto"})  # default
```

### model.hf_text.max_text_len
Maximum text length. If > 0, uses minimum of this value and model's max length. If ≤ 0, uses model's maximum.
```python
predictor.fit(hyperparameters={"model.hf_text.max_text_len": 512})  # default
predictor.fit(hyperparameters={"model.hf_text.max_text_len": -1})   # use tokenizer max
```

### model.hf_text.insert_sep
Insert SEP token between texts from different dataframe columns.
```python
predictor.fit(hyperparameters={"model.hf_text.insert_sep": True})  # default
```

### model.hf_text.text_segment_num
Number of text segments in token sequence. Each segment has one token type ID. Uses minimum of this value and model default.
```python
predictor.fit(hyperparameters={"model.hf_text.text_segment_num": 2})  # default
```

## Text Processing

### model.hf_text.stochastic_chunk
Randomly cut text chunk when tokens exceed `max_text_len`. If False, cuts from index 0.
```python
predictor.fit(hyperparameters={"model.hf_text.stochastic_chunk": False})  # default
predictor.fit(hyperparameters={"model.hf_text.stochastic_chunk": True})   # random chunk
```

### model.hf_text.text_aug_detect_length
Minimum token count to enable text augmentation.
```python
predictor.fit(hyperparameters={"model.hf_text.text_aug_detect_length": 10})  # default
```

### model.hf_text.text_trivial_aug_maxscale
Maximum percentage of tokens for augmentation. Randomly samples percentage in [0, maxscale] and applies one of four operations: synonym replacement, word swap, word deletion, or punctuation insertion.
```python
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0})    # default: disabled
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0.1})  # enable augmentation
```

### model.hf_text.gradient_checkpointing
Enable gradient checkpointing to reduce memory consumption during gradient calculation.
```python
predictor.fit(hyperparameters={"model.hf_text.gradient_checkpointing": False})  # default
predictor.fit(hyperparameters={"model.hf_text.gradient_checkpointing": True})   # enable
```

# FT-Transformer Configuration

### model.ft_transformer.checkpoint_name
Initialize FT-Transformer from local or remote pre-trained weights.
```python
predictor.fit(hyperparameters={"model.ft_transformer.checkpoint_name": None})  # default: no pretrained weights

# local checkpoint
predictor.fit(hyperparameters={"model.ft_transformer.checkpoint_name": 'my_checkpoint.ckpt'})

# remote checkpoint
predictor.fit(hyperparameters={"model.ft_transformer.checkpoint_name": 'https://automl-mm-bench.s3.amazonaws.com/ft_transformer_pretrained_ckpt/iter_2k.ckpt'})
```

### model.ft_transformer.num_blocks
Number of transformer blocks.
```python
predictor.fit(hyperparameters={"model.ft_transformer.num_blocks": 3})  # default
```

### model.ft_transformer.token_dim
Token dimension after categorical/numerical tokenization.
```python
predictor.fit(hyperparameters={"model.ft_transformer.token_dim": 192})  # default
```

### model.ft_transformer.hidden_size
Model embedding dimension.
```python
predictor.fit(hyperparameters={"model.ft_transformer.hidden_size": 192})  # default
```

### model.ft_transformer.ffn_hidden_size
Feed-Forward Network hidden layer dimension. **Note**: Standard Transformer uses 4× model hidden size; AutoMM defaults to 1× (equal to model hidden size).
```python
predictor.fit(hyperparameters={"model.ft_transformer.ffn_hidden_size": 192})  # default
```

# Image Model Configuration

### model.timm_image.checkpoint_name
Select TIMM image backbone.
```python
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "swin_base_patch4_window7_224"})  # default
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "vit_base_patch32_224"})          # ViT base
```

### model.timm_image.train_transforms
Training image augmentations. Accepts:
- **List of strings**: `resize_to_square`, `resize_shorter_side`, `center_crop`, `random_resize_crop`, `random_horizontal_flip`, `random_vertical_flip`, `color_jitter`, `affine`, `randaug`, `trivial_augment`
- **List of callable objects**: e.g., torchvision transforms (must be pickle-able)

```python
# default
predictor.fit(hyperparameters={"model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"]})

# custom string transforms
predictor.fit(hyperparameters={"model.timm_image.train_transforms": ["random_resize_crop", "random_horizontal_flip"]})

# torchvision transforms
predictor.fit(hyperparameters={"model.timm_image.train_transforms": [
    torchvision.transforms.RandomResizedCrop(224), 
    torchvision.transforms.RandomHorizontalFlip()
]})
```

# Image Model Configuration (Continued)

### model.timm_image.val_transforms
Validation/test/deployment image transforms. Accepts list of strings or callable pickle-able objects.
```python
predictor.fit(hyperparameters={"model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"]})  # default
predictor.fit(hyperparameters={"model.timm_image.val_transforms": ["resize_to_square"]})
predictor.fit(hyperparameters={"model.timm_image.val_transforms": [torchvision.transforms.Resize((224, 224))]})
```

## MMDetection Configuration

### model.mmdet_image.checkpoint_name
MMDetection model selection. **Use "yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", or "yolox_x" for AutoGluon-compatible YOLOX models.**
```python
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_8xb24-320-300e_coco"})  # default
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "yolox_l"})
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "dino-5scale_swin-l_8xb2-36e_coco"})
```

### model.mmdet_image.output_bbox_format
Bounding box output format:
- `"xyxy"`: [x1, y1, x2, y2] - corners format (default)
- `"xywh"`: [x1, y1, w, h] - corner + width/height format

```python
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.output_bbox_format": "xyxy"})  # default
```

### model.mmdet_image.frozen_layers
Layers to freeze during training. All layers containing specified substrings will be frozen.
```python
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.frozen_layers": []})  # default: freeze nothing
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.frozen_layers": ["backbone"]})
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.frozen_layers": ["backbone", "neck"]})
```

## SAM (Segment Anything Model) Configuration

### model.sam.checkpoint_name
Huggingface SAM backbone selection.
```python
predictor.fit(hyperparameters={"model.sam.checkpoint_name": "facebook/sam-vit-huge"})   # default
predictor.fit(hyperparameters={"model.sam.checkpoint_name": "facebook/sam-vit-large"})
predictor.fit(hyperparameters={"model.sam.checkpoint_name": "facebook/sam-vit-base"})
```

### model.sam.train_transforms
Training augmentations. Currently supports `random_horizontal_flip`.
```python
predictor.fit(hyperparameters={"model.sam.train_transforms": ["random_horizontal_flip"]})  # default
```

### model.sam.img_transforms
Input image processing. Currently supports `resize_to_square`.
```python
predictor.fit(hyperparameters={"model.sam.img_transforms": ["resize_to_square"]})  # default
```

### model.sam.gt_transforms
Ground truth mask processing. Currently supports `resize_gt_to_square`.
```python
predictor.fit(hyperparameters={"model.sam.gt_transforms": ["resize_gt_to_square"]})  # default
```

### model.sam.frozen_layers
Modules to freeze during training.
```python
predictor.fit(hyperparameters={"model.sam.frozen_layers": ["mask_decoder.iou_prediction_head", "prompt_encoder"]})  # default
```

### model.sam.num_mask_tokens
Number of mask proposals from SAM's mask decoder.
```python
predictor.fit(hyperparameters={"model.sam.num_mask_tokens": 1})  # default
```

### model.sam.ignore_label
Target value to ignore in loss and metric calculations.
```python
predictor.fit(hyperparameters={"model.sam.ignore_label": 255})  # default
```

# Data Configuration

## Image Data

### data.image.missing_value_strategy
Handling for missing/failed images:
- `"skip"`: skip samples with missing images
- `"zero"`: replace with zero image (default)

```python
predictor.fit(hyperparameters={"data.image.missing_value_strategy": "zero"})  # default
```

## Text Data

### data.text.normalize_text
Normalize text with encoding problems through encoding/decoding series.
```python
predictor.fit(hyperparameters={"data.text.normalize_text": False})  # default
predictor.fit(hyperparameters={"data.text.normalize_text": True})
```

## Categorical & Numerical Data

### data.categorical.convert_to_text
Treat categorical data as text. If True, disables categorical models (`categorical_mlp`, `categorical_transformer`).
```python
predictor.fit(hyperparameters={"data.categorical.convert_to_text": False})  # default
```

### data.numerical.convert_to_text
Convert numerical data to text. If True, disables numerical models (`numerical_mlp`, `numerical_transformer`).
```python
predictor.fit(hyperparameters={"data.numerical.convert_to_text": False})  # default
```

### data.numerical.scaler_with_mean
Center numerical data (excluding labels) before scaling.
```python
predictor.fit(hyperparameters={"data.numerical.scaler_with_mean": True})  # default
```

### data.numerical.scaler_with_std
Scale numerical data (excluding labels) to unit variance.
```python
predictor.fit(hyperparameters={"data.numerical.scaler_with_std": True})  # default
```

## Label Processing

### data.label.numerical_preprocessing
Numerical label processing for regression:
- `"standardscaler"`: remove mean, scale to unit variance (default)
- `"minmaxscaler"`: scale to range (0, 1)

```python
predictor.fit(hyperparameters={"data.label.numerical_preprocessing": "standardscaler"})  # default
```

### data.pos_label
Positive label for binary classification. **Required for metrics like roc_auc, average_precision, and f1.**
```python
predictor.fit(hyperparameters={"data.pos_label": None})  # default
predictor.fit(hyperparameters={"data.pos_label": "changed"})  # example
```

### data.column_features_pooling_mode
Aggregate multi-column features. **Currently works only for `few_shot_classification`.**
- `"concat"`: concatenate column features
- `"mean"`: average column features (constant dimension)

```python

# Data Configuration (Continued)

### data.column_features_pooling_mode (continued)
```python
predictor.fit(hyperparameters={"data.column_features_pooling_mode": "concat"})  # default
predictor.fit(hyperparameters={"data.column_features_pooling_mode": "mean"})
```

## Mixup & Cutmix Augmentation

### data.mixup.turn_on
Enable Mixup augmentation during training.
```python
predictor.fit(hyperparameters={"data.mixup.turn_on": False})  # default
predictor.fit(hyperparameters={"data.mixup.turn_on": True})
```

### data.mixup.mixup_alpha
Mixup alpha parameter. Mixup is active when > 0. Set to 1.0 to disable.
```python
predictor.fit(hyperparameters={"data.mixup.mixup_alpha": 0.8})  # default
predictor.fit(hyperparameters={"data.mixup.mixup_alpha": 1.0})  # disable
```

### data.mixup.cutmix_alpha
Cutmix alpha parameter. Cutmix is active when > 0.
```python
predictor.fit(hyperparameters={"data.mixup.cutmix_alpha": 1.0})  # default: disabled
predictor.fit(hyperparameters={"data.mixup.cutmix_alpha": 0.8})  # enable
```

### data.mixup.prob
Probability of applying Mixup or Cutmix when enabled.
```python
predictor.fit(hyperparameters={"data.mixup.prob": 1.0})  # default
```

### data.mixup.switch_prob
Probability of switching to Cutmix instead of Mixup when both are active.
```python
predictor.fit(hyperparameters={"data.mixup.switch_prob": 0.5})  # default
```

### data.mixup.mode
Application scope for Mixup/Cutmix parameters:
- `"batch"`: per batch (default)
- `"pair"`: per pair of elements
- `"elem"`: per element

```python
predictor.fit(hyperparameters={"data.mixup.mode": "batch"})  # default
```

### data.mixup.label_smoothing
Label smoothing applied to mixed label tensors.
```python
predictor.fit(hyperparameters={"data.mixup.label_smoothing": 0.1})  # default
```

### data.mixup.turn_off_epoch
Epoch at which to stop Mixup/Cutmix.
```python
predictor.fit(hyperparameters={"data.mixup.turn_off_epoch": 5})  # default
```

# Distillation Configuration

## Knowledge Distillation Parameters

### distiller.soft_label_loss_type
Loss function for teacher-to-student supervision using teacher's logits.
```python
predictor.fit(hyperparameters={"distiller.soft_label_loss_type": "cross_entropy"})  # default: classification
predictor.fit(hyperparameters={"distiller.soft_label_loss_type": "mse"})            # default: regression
```

### distiller.temperature
Temperature scaling for teacher and student logits before computing soft label loss: `teacher_logits / temperature`, `student_logits / temperature`.
```python
predictor.fit(hyperparameters={"distiller.temperature": 5})  # default: classification
predictor.fit(hyperparameters={"distiller.temperature": 1})
```

### distiller.hard_label_weight
Weight for student's hard label (ground truth) loss: `hard_label_loss * hard_label_weight`.
```python
predictor.fit(hyperparameters={"distiller.hard_label_weight": 0.2})  # default: classification
predictor.fit(hyperparameters={"distiller.hard_label_weight": 1})    # no scaling
```

### distiller.soft_label_weight
Weight for student's soft label (teacher output) loss: `soft_label_loss * soft_label_weight`.
```python
predictor.fit(hyperparameters={"distiller.soft_label_weight": 50})  # default: classification
predictor.fit(hyperparameters={"distiller.soft_label_weight": 1})   # no scaling
```