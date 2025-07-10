# Condensed: Pretraining

Summary: This tutorial covers efficient pretraining techniques for large video-text datasets like Howto100M, focusing on data sharding implementation for memory optimization through memory-mapped shards. It explains how to run MFM+MLM pretraining on multi-GPU setups and details VideoCLIP retrieval model implementation using faiss. Key functionalities include video representation through clip pooling, memory-efficient data processing with `Sharded*` processors, and retrieval task implementation in `mmpt/tasks/retritask.py`. The tutorial provides practical command-line instructions and configuration files for both standard pretraining and retrieval-based pretraining workflows.

*This is a condensed version that preserves essential implementation details and context.*

# Pretraining

## Data Sharding
Pretraining on Howto100M requires optimized preprocessing due to millions of videos/captions that can't fit in memory:

- We use data sharding to pack multiple videos into shards for both videos and captions
- Shards are memory-mapped to reduce IO access frequency
- Use processors starting with `Sharded*`
- Default config: `projects/task/how2.yaml`

## Training Requirements
- One or multiple nodes
- Each node: 8 GPUs with 32GB memory
- Launch MFM+MLM pretraining:
  ```bash
  python locallaunch.py projects/mfmmlm/how2.yaml
  ```

## Pre-training with Retrieval Model (VideoCLIP)
```bash
# Install required dependency
conda install faiss-cpu -c pytorch
```

Implementation details:
- Built on video hidden states and faiss
- Video representation: average of 8 clips of pooled visual/text hidden states
- Implementation in `mmpt/tasks/retritask.py`
- Config file: `projects/retri/videoretri.yaml`