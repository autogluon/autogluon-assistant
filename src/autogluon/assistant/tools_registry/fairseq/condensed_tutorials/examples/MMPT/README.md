# Condensed: VideoCLIP and VLM

Summary: This tutorial provides implementation details for VideoCLIP and VLM, two multimodal video understanding models. It covers installation procedures, model loading, and inference with pre-trained checkpoints. Key functionalities include zero-shot transfer learning, contrastive learning for video-text alignment, and masked language modeling for multimodal understanding. The code demonstrates how to process video frames with text inputs, run inference to get similarity scores, and includes training/evaluation pipelines with configuration examples. The tutorial highlights the modular processor system for data handling and memory optimization techniques for multi-GPU training, making it valuable for implementing video-text multimodal models.

*This is a condensed version that preserves essential implementation details and context.*

# VideoCLIP and VLM Implementation Guide

## Overview
This toolkit implements two multimodal video understanding models:
- **VideoCLIP**: Contrastive learning model for zero-shot transfer
- **VLM**: Masked language model style pre-training with masked modality model

## Installation

```bash
# Install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .
export MKL_THREADING_LAYER=GNU

# Install MMPT
cd examples/MMPT
pip install -e .
```

**Requirements**:
- Python 3.8.8, PyTorch 1.8/1.9, CUDA 11.0
- `transformers==3.4` (API compatibility requirement)
- Some tasks require pandas: `conda install pandas`

## Checkpoints and Models

1. Download pre-trained S3D models:
   - Place at `pretrained_models/s3d_dict.npy` and `pretrained_models/s3d_howto100m.pth`

2. Download model checkpoints:
   - VideoCLIP: `https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt` → `runs/retri/videoclip`
   - VLM: `https://dl.fbaipublicfiles.com/MMPT/mtm/vlm/checkpoint_best.pt` → `runs/mtm/vlm`

## Inference Example

```python
import torch
from mmpt.models import MMPTModel

# Load model
model, tokenizer, aligner = MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")
model.eval()

# Prepare inputs
video_frames = torch.randn(1, 2, 30, 224, 224, 3)  # B, T, FPS, H, W, C (30 fps for s3d)
caps, cmasks = aligner._build_text_seq(
    tokenizer("some text", add_special_tokens=False)["input_ids"]
)
caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

# Run inference
with torch.no_grad():
    output = model(video_frames, caps, cmasks, return_score=True)
print(output["score"])  # dot-product
```

## Training and Evaluation

Generate configs for all stages:
```bash
python locallaunch.py projects/retri/videoclip.yaml --dryrun
```

### Zero-shot evaluation:
```bash
python locallaunch.py projects/retri/videoclip/test_youcook_zs.yaml --jobtype local_predict
```

### Fine-tuning:
```bash
# Check command then run (uses 2 GPUs as in paper)
python locallaunch.py projects/retri/videoclip/youcook_videoclip.yaml --jobtype local_single --dryrun
```

### Testing fine-tuned model:
```bash
python locallaunch.py projects/retri/videoclip/test_youcook_videoclip.yaml --jobtype local_predict
```

### Pre-training:
```bash
# Check command then run (paper used 8 GPUs with local_big)
python locallaunch.py projects/retri/videoclip/how2.yaml --jobtype local_single --dryrun
```

## Key Components

### Processors
The toolkit uses a modular processor system for data handling:
- **MetaProcessor**: Loads dataset metadata (video IDs)
- **VideoProcessor**: Loads video features (e.g., S3D features)
- **TextProcessor**: Loads text features (e.g., BERT pre-tokenized clips)
- **Aligner**: Core class that prepares training data (clip sampling, token masking)

### Performance Optimizations
- Uses `ShardedTensor` for efficient memory management
- Enables near random access to features stored in continuous disk space
- Reduces I/O load for multi-GPU training

## Configuration
All training/testing pipelines are organized in YAML config files under the `projects` directory:
- VideoCLIP: `projects/retri/videoclip.yaml`
- VLM: `projects/mtm/vlm.yaml`