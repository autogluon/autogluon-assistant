# Condensed: 

Summary: This tutorial presents a speech resynthesis implementation that disentangles speech into three discrete representations: content, prosody, and speaker identity. The code enables controllable speech synthesis and ultra-lightweight speech coding (365 bits/second). Key functionalities include speech content extraction, prosodic information modeling, and speaker identity preservation. The implementation is evaluated on F0 reconstruction, speaker identification, speech intelligibility, and overall quality. This resource would help with implementing speech synthesis systems, voice conversion applications, and creating compact speech representations for low-bandwidth communication.

*This is a condensed version that preserves essential implementation details and context.*

# Speech Resynthesis from Discrete Disentangled Self-Supervised Representations

## Implementation Overview

This project implements speech resynthesis using disentangled self-supervised representations that separately extract:
- Speech content
- Prosodic information
- Speaker identity

These discrete representations enable controllable speech synthesis and can be used for ultra-lightweight speech coding at 365 bits per second.

## Key Components

![Architecture diagram](img/fig.png)

## Technical Resources
- [GitHub Repository](https://github.com/facebookresearch/speech-resynthesis)
- [Paper](https://arxiv.org/pdf/2104.00355.pdf)
- [Audio Samples](https://speechbot.github.io/resynthesis/index.html)

## Performance Metrics
The system is evaluated on:
- F0 reconstruction
- Speaker identification performance (resynthesis and voice conversion)
- Speech intelligibility
- Overall quality via subjective human evaluation

## Citation
```
@inproceedings{polyak21_interspeech,
  author={Adam Polyak and Yossi Adi and Jade Copet and 
          Eugene Kharitonov and Kushal Lakhotia and 
          Wei-Ning Hsu and Abdelrahman Mohamed and Emmanuel Dupoux},
  title={{Speech Resynthesis from Discrete Disentangled Self-Supervised Representations}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
}
```