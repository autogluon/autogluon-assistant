# Condensed: Benchmarking

Summary: This tutorial presents a benchmarking framework for speech-to-speech translation (S2ST) models, measuring runtime, memory usage, and FLOPS. It implements core inference modules from fairseq for both end-to-end and cascaded models, focusing solely on model inference costs while excluding data processing overhead. The framework runs on CPU environments and supports two dataset formats: NPY datasets with preprocessed tensors and raw audio datasets. This resource helps developers evaluate S2ST model performance metrics, compare different architectures, and optimize inference efficiency in production environments.

*This is a condensed version that preserves essential implementation details and context.*

# Benchmarking Framework for S2ST Models

## Overview

This framework benchmarks speech-to-speech translation (S2ST) models on:
- **Runtime**: Average inference time per example (using `timeit`)
- **Max memory**: Maximum memory usage in MiB (using `memory_profiler`)
- **FLOPS**: Average floating point operations per example (using PAPI library)

The framework supports both end-to-end and cascaded models, ensuring fair comparison by measuring only model inference costs while ignoring intermediate data processing.

## Implementation Details

- Core inference modules reimplemented from `fairseq_cli/generate.py` and `examples/speech_to_text/generate_waveform.py`
- All benchmarking runs on CPU (production environment standard)
- For cascaded models, max memory is determined by finding the maximum across all stages

## Usage

```python
CUBLAS_WORKSPACE_CONFIG=:4096:8 python examples/speech_to_speech/benchmarking/get_metrics.py '' --config $config
```

## Dataset Formats

1. **NPY dataset**: List of samples saved as `.npy` file:
   ```python
   sample = {
       "id": xx,
       "net_input": {
           "src_tokens": torch.tensor([]),
           "src_lengths": torch.tensor([])
       }
   }
   ```

2. **Raw dataset**: List of raw audio paths (similar to wav2vec2 input TSV file)