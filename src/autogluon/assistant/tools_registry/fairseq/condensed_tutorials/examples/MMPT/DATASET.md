# Condensed: Dataset

Summary: This tutorial covers video-text dataset processing for multimodal models, focusing on S3D feature extraction for videos. It details implementation techniques for handling the Howto100M dataset, including caption cleaning, deduplication, and efficient storage with ShardedTensor. The code helps with video feature extraction, text tokenization, and dataset preparation tasks. Key functionalities include the PathBuilder class for tracking video IDs, fp16 feature storage for efficiency, and processing steps for both Howto100M and other datasets (Youcook, MSRVTT) with their specific configurations and extraction workflows.

*This is a condensed version that preserves essential implementation details and context.*

# Dataset Processing for Video-Text Models

## Video Feature Extraction with S3D

We use pre-trained [S3D](https://github.com/antoine77340/S3D_HowTo100M) for video feature extraction.

### Setup Requirements
```bash
# Place models in these locations
pretrained_models/s3d_dict.npy
pretrained_models/s3d_howto100m.pth

# Install dependencies
conda install -c anaconda pandas
pip install ffmpeg-python
```

The `PathBuilder` class automatically tracks video IDs and maps source video paths to feature locations.

## Howto100M Dataset Processing

Key preprocessing differences from existing implementations:
1. Using `raw_caption.json` instead of `caption.json` for pure self-supervision
2. Removing partially duplicated texts via `mmpt/processors/dedupprocessor.py`
3. Sharding video/text features with `SharedTensor` for faster loading

### Processing Steps

#### Video Processing
```bash
# Extract video features
bash scripts/video_feature_extractor/how2/s3d.sh

# Features stored in fp16 by default to save space and speed up training
# Video IDs are split into:
data/how2/how2_s3d_train.lst
data/how2/how2_s3d_val.lst

# Pack features into ShardedTensor
python scripts/video_feature_extractor/shard_feature.py
```

#### Text Processing
```bash
# Clean captions
python -m mmpt.processors.dedupprocessor

# Tokenize cleaned captions
python scripts/text_token_extractor/pretokenization.py scripts/text_token_extractor/configs/bert-base-uncased.yaml
```

## Other Datasets (Youcook, MSRVTT)

We use the versions from Howto100M and MILNCE projects:
- Download data to `data/youcook` and `data/msrvtt`
- Configuration details in `projects/task/youcook.yaml` and `projects/task/vtt.yaml`
- Extract features similar to Howto100M's first step
- Text is read from metadata directly with on-the-fly tokenization