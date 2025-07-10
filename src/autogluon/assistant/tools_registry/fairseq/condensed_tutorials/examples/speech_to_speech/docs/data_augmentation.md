# Condensed: Noise and audio augmentation techniques

Summary: This tutorial covers audio augmentation techniques for speech processing in Fairseq. It details implementation of four noise augmentation transforms (music, babble, sporadic, and background noise) and two dataset-level transforms (utterance concatenation and noisy overlap). Each transform includes configuration parameters, implementation details, and benchmark results showing their effectiveness on different noise conditions. The tutorial explains how to integrate these transforms into Fairseq's audio processing pipeline, categorizing them as AudioFeatureTransform, AudioWaveformTransform, or AudioDatasetTransform. It also provides guidance on creating custom transforms with practical code examples for handling batch-level and dataset-level transformations, making it valuable for implementing robust speech recognition systems.

*This is a condensed version that preserves essential implementation details and context.*

# Noise and Audio Augmentation Techniques

## Transform Types
All transforms are subclasses of:
- `AudioFeatureTransform`
- `AudioWaveformTransform`
- `AudioDatasetTransform`

## In-built Transforms

### 1. Utterance Concatenation (`concataugment`)
An `AudioDatasetTransform` that combines samples with probability `rate`.

```python
# Configuration parameters
{
    "rate": 0.25,           # Probability of concatenation
    "max_tokens": 300,      # Maximum tokens in concatenated sequence
    "attempts": 5           # Maximum attempts before skipping concatenation
}
```

**Warning**: Watch for OOMs - use smaller batch sizes as needed.

### 2. Noise Augmentation Suite

#### 2.1 Music Augmentation (`musicaugment`)
An `AudioWaveformTransform` that overlays music on speech samples.

```python
# Configuration parameters
{
    "samples_path": "/path/to/music/files",  # Required
    "rate": 0.25,                            # Probability of applying augmentation
    "snr_min": 5,                            # Minimum signal-to-noise ratio
    "snr_max": 15                            # Maximum signal-to-noise ratio
}
```

#### 2.2 Babble Augmentation (`babbleaugment`)
An `AudioWaveformTransform` that overlays 3-7 speech files as background noise.

```python
# Configuration parameters
{
    "samples_path": "/path/to/speech/files",  # Required
    "rate": 0.25,                             # Probability of applying augmentation
    "snr_min": 5,                             # Minimum signal-to-noise ratio
    "snr_max": 15                             # Maximum signal-to-noise ratio
}
```

#### 2.3 Sporadic Noise Augmentation (`sporadicnoiseaugment`)
An `AudioWaveformTransform` that adds intermittent noise clips.

```python
# Configuration parameters
{
    "samples_path": "/path/to/noise/files",  # Required
    "rate": 0.25,                            # Probability of applying augmentation
    "snr_min": 5,                            # Minimum signal-to-noise ratio
    "snr_max": 15,                           # Maximum signal-to-noise ratio
    "noise_rate": 0.5,                       # Noises per second
    "noise_len_mean": 0.2,                   # Mean noise clip length (seconds)
    "noise_len_std": 0.1                     # Standard deviation of noise length
}
```

#### 2.4 Background Noise Augmentation (`backgroundnoiseaugment`)
An `AudioWaveformTransform` that adds continuous background noise.

```python
# Configuration parameters
{
    "samples_path": "/path/to/noise/files",  # Required
    "rate": 0.25,                            # Probability of applying augmentation
    "snr_min": 5,                            # Minimum signal-to-noise ratio
    "snr_max": 15                            # Maximum signal-to-noise ratio
}
```

All noise augmentation methods overlay noise with a signal-to-noise ratio randomly selected from a uniform distribution between `snr_min` and `snr_max`.

# Mixed Babble and Background Noise Augmentation

## NoisyOverlapAugment

This augmentation technique is based on Algorithm 1 in [WavLM paper](https://arxiv.org/abs/2110.13900). It combines background noise with another audio sample from the batch.

**Key Implementation Details:**
- The noise track length is randomly chosen between 0 and half of the original sample length
- The noise can be either another audio sample from the batch or a background noise track

**Configuration Parameters:**
```python
# NoisyOverlapAugment parameters
noises_path = "/path/to/noise/files"  # Required, no default
rate = 0.25  # Probability of applying augmentation
mixing_noise_rate = 0.1  # Probability of using background noise vs. batch sample
noise_snr_min = -5  # Min SNR for background noise
noise_snr_max = 5   # Max SNR for background noise
utterance_snr_min = -5  # Min SNR for batch sample mixing
utterance_snr_max = 5   # Max SNR for batch sample mixing
```

## Benchmark Results

### Clean Data Evaluation
| Augmentation | Training loss | BLEU (covost) | BLEU (epst) | BLEU (mtedx) |
|-------------|--------------|--------------|------------|------------|
| None | 3.954 | 24.984 | 23.962 | 24.448 |
| ConcatAugment | 3.940 | 25.322 | 26.124 | 26.19 |
| BabbleAugment | 3.957 | 24.226 | 23.186 | 22.368 |
| MusicAugment | 3.954 | 25.096 | 24.301 | 23.341 |
| NoisyOverlapAugment | 3.954 | 24.949 | 24.015 | 23.768 |

### Music Noise Evaluation (SNR = -5 to 5)
| Augmentation | BLEU (covost) | BLEU (epst) | BLEU (mtedx) |
|-------------|--------------|------------|------------|
| None | 15.785 | 21.105 | 16.944 |
| MusicAugment | 20.345 | 23.126 | 19.433 |
| Combined Augmentations | 19.724 | 22.659 | 17.852 |

### Babble Noise Evaluation (SNR = -5 to 5)
| Augmentation | BLEU (covost) | BLEU (epst) | BLEU (mtedx) |
|-------------|--------------|------------|------------|
| None | 4.092 | 13.514 | 5.13 |
| BabbleAugment | 16.12 | 21.097 | 13.996 |
| Combined Augmentations | 14.692 | 20.882 | 14.45 |

### NoisyOverlap Evaluation
| Augmentation | BLEU (covost) | BLEU (epst) | BLEU (mtedx) |
|-------------|--------------|------------|------------|
| None | 21.245 | 22.24 | 20.994 |
| NoisyOverlapAugment | 23.371 | 23.396 | 22.627 |
| Combined Augmentations | 22.206 | 22.414 | 21.375 |

**Key Findings:**
- Each augmentation technique performs best when evaluated on test data with the same type of noise
- ConcatAugment consistently improves performance on clean data
- Combined augmentations provide robustness across different noise types
- NoisyOverlapAugment shows strong performance when evaluated on data with similar noise characteristics

# Using Transforms in Fairseq Audio Processing

## Transform Types and Configuration

Transforms are categorized into three types, each requiring specific configuration:

1. **Dataset Transforms** (`AudioDatasetTransform`): Listed under `dataset_transforms`
   - Examples: `concataugment`, `noisyoverlapaugment`

2. **Waveform Transforms** (`AudioWaveformTransform`): Listed under `waveform_transforms`
   - Examples: `musicaugment`, `babbleaugment`, `sporadicnoiseaugment`, `backgroundnoiseaugment`

3. **Feature Transforms** (`AudioFeatureTransform`): Listed under `feature_transforms`

You can apply transforms conditionally using flags like `_train` or `_eval`:

```yaml
# Example: Music augmentation for training only
musicaugment:
  samples_path: ${MUSIC_PATH}
  snr_min: 10 
  snr_max: 15
  rate: 0.25
waveform_transforms:
  _train:
  - musicaugment
```

```yaml
# Example: Concatenation augmentation
concataugment:
  rate: 0.25
  max_tokens: 3000
  attempts: 5
dataset_transforms:
  _train:
  - concataugment
```

Multiple transforms can be combined:

```yaml
musicaugment:
  samples_path: ${MUSIC_PATH}
  snr_min: 5 
  snr_max: 20
  rate: 0.25
backgroundnoiseaugment:
  samples_path: ${NOISES_PATH}
  snr_min: 10
  snr_max: 20
  rate: 0.1
sporadicnoiseaugment:
  samples_path: ${NOISES_PATH}
  snr_min: 5
  snr_max: 15
  rate: 0.1
  noise_rate: 0.25
waveform_transforms:
  _train:
  - musicaugment
  - backgroundnoiseaugment
  - sporadicnoiseaugment
```

## Creating Custom Transforms

### Step 1: Choose the Right Base Class

- **AudioFeatureTransform**: For transforms applied to spectrograms
- **AudioWaveformTransform**: For transforms applied to audio waveforms
- **AudioDatasetTransform**: For transforms requiring multiple dataset items (e.g., concatenation)

### Step 2: Basic Transform Setup

```python
# Example for a dataset transform
from fairseq.data.audio.dataset_transforms import (
  AudioDatasetTransform,
  register_audio_dataset_transform
)

@register_audio_dataset_transform("concataugment")
class ConcatAugment(AudioDatasetTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return ConcatAugment(
            _config.get("rate", _DEFAULTS["rate"]),
            _config.get("max_tokens", _DEFAULTS["max_tokens"]),
            _config.get("attempts", _DEFAULTS["attempts"]),
        )
    
    def __init__(
        self,
        rate=_DEFAULTS["rate"],
        max_tokens=_DEFAULTS["max_tokens"],
        attempts=_DEFAULTS["attempts"],
    ):
        self.rate, self.max_tokens, self.attempts = rate, max_tokens, attempts
    
    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join([
                f"rate={self.rate}",
                f"max_tokens={self.max_tokens}",
                f"attempts={self.attempts}",
            ])
            + ")"
        )
```

### Step 3: Implementing Transform Logic

#### For Feature Transforms
Implement `__call__` to process spectrograms:

```python
def __call__(self, x):
    # x is a spectrogram (np.ndarray)
    x = np.subtract(x, self.mean)
    x = np.divide(x, self.std)
    return x
```

#### For Waveform Transforms
Implement `__call__` to process audio waveforms:

```python
def __call__(self, source, sample_rate=None):
    # source is audio waveform (channels x length)
    if np.random.random() > self.rate:
        return source

    noise = self._get_noise(
        source.shape, always_2d=True, use_sample_rate=sample_rate
    )
    return self._mix(source, noise, rand_uniform(self.snr_min, self.snr_max)), sample_rate
```

#### For Dataset Transforms
These require direct integration into `fairseq/data/audio/speech_to_text_dataset.py` using:
1. Check if transform exists: `self.dataset_transforms.has_transform(TRANSFORM_CLS)`
2. Apply transform: `self.dataset_transforms.get_transform(TRANSFORM_CLS)`

# Implementing Complex Data Transforms in Fairseq

## Batch-Level Transforms: NoisyOverlapAugment

This transform requires access to multiple items within the same batch simultaneously.


...(truncated)