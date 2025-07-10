# Condensed: Zero-shot Transfer and Finetuning

Summary: This tutorial covers implementation techniques for multimodal video-text tasks, focusing on zero-shot transfer and finetuning. It explains how to configure datasets for text-video retrieval, video captioning, and video QA using specific processors defined in `mmpt.processors.dsprocessor`. The tutorial details how to run pre-trained models in zero-shot settings, perform finetuning with checkpoint restoration, and execute testing with appropriate predictors and metrics. It includes code examples for dataset configuration, finetuning setup, and testing procedures, along with information about required third-party libraries for specific tasks like Youcook captioning and CrossTask.

*This is a condensed version that preserves essential implementation details and context.*

# Zero-shot Transfer and Finetuning

## Dataset Support

Currently supported datasets and tasks:
- **Text-video retrieval**: MSRVTT, Youcook, DiDeMo
- **Video captioning**: Youcook
- **Video QA**: MSRVTT-QA

## Implementation Details

### Dataset Configuration
All finetuning datasets are defined in `mmpt.processors.dsprocessor`. Each task has specific processors:
```
# Configure dataset in config file
# Example from projects/task/vtt.yaml
dataset:
  video_processor: ...
  text_processor: ...
  aligner: ...
```

### Zero-shot Transfer
Run pre-trained models directly on test data using configs with pattern `projects/task/*_zs_*.yaml`.

### Fine-tuning
Similar to pretraining but requires specifying:
```yaml
# From projects/task/ft.yaml
fairseq:
  checkpoint:
    restore_file: /path/to/checkpoint.pt
  # Reset optimizers and other training parameters
```

Typical finetuning setup:
```
# Run on 2 GPUs
python locallaunch.py projects/task/vtt.yaml
```

### Testing
1. Define testing config (e.g., `projects/task/test_vtt.yaml`)
2. Specify appropriate predictor and metric:
   ```yaml
   predictor: RetrievalPredictor  # For retrieval tasks
   metric: RetrievalMetric        # Task-specific metric
   ```
3. Run testing:
   ```
   python locallaunch.py projects/mfmmlm/test_vtt.yaml
   ```

## Required Third-party Libraries

- **Youcook captioning**: `https://github.com/Maluuba/nlg-eval`
- **CrossTask**: Requires `https://github.com/DmZhukov/CrossTask`'s `dp` module
  ```
  # Install CrossTask dp module
  cd third-party/CrossTask
  python setup.py build_ext --inplace
  ```