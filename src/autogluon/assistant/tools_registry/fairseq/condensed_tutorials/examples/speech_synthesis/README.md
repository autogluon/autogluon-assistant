# Condensed: Speech Synthesis (S^2)

Summary: This tutorial introduces fairseq S^2, a speech synthesis toolkit that implements both autoregressive and non-autoregressive models for text-to-speech tasks. It covers techniques for multi-speaker synthesis, audio preprocessing for handling less curated data, and automatic evaluation metrics. The tutorial provides implementation examples for single-speaker synthesis on LJSpeech and multi-speaker synthesis on VCTK and Common Voice datasets. Developers can leverage this toolkit for building speech synthesis systems with features like denoising, voice activity detection (VAD), and compatibility with speech-to-text configurations, making it valuable for various TTS applications.

*This is a condensed version that preserves essential implementation details and context.*

# Speech Synthesis (S^2)

## Key Features
- Autoregressive and non-autoregressive models
- Multi-speaker synthesis
- Audio preprocessing (denoising, VAD) for less curated data
- Automatic metrics for model development
- Compatible data configuration with S2T

## Implementation Examples
- [Single-speaker synthesis on LJSpeech](docs/ljspeech_example.md)
- [Multi-speaker synthesis on VCTK](docs/vctk_example.md)
- [Multi-speaker synthesis on Common Voice](docs/common_voice_example.md)

## Citation
```
@article{wang2021fairseqs2,
  title={fairseq S\^{} 2: A Scalable and Integrable Speech Synthesis Toolkit},
  author={Wang, Changhan and Hsu, Wei-Ning and Adi, Yossi and Polyak, Adam and Lee, Ann and Chen, Peng-Jen and Gu, Jiatao and Pino, Juan},
  journal={arXiv preprint arXiv:2109.06912},
  year={2021}
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```