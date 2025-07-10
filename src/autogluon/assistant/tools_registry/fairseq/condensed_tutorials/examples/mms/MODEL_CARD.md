# Condensed: MMS Model Card

Summary: This tutorial introduces Meta's MMS (Massively Multilingual Speech) models, covering implementation details for transformer-based speech processing across 1000+ languages. It provides guidance on using pre-trained (300M/1B parameters) and fine-tuned variants for automatic speech recognition, language identification, and speech synthesis tasks. The tutorial details the model architecture, training datasets (including VoxPopuli, MLS, CommonVoice), evaluation metrics (CER, WER), and implementation considerations. It's particularly valuable for developers implementing multilingual speech processing systems, with specific information on model variants, data requirements, and performance characteristics across different languages.

*This is a condensed version that preserves essential implementation details and context.*

# MMS Model Card - Condensed Implementation Guide

## Model Overview
- **Developer**: FAIR team
- **Versions**: 300M and 1B parameter variants
- **Architecture**: Transformer-based speech model
- **License**: CC BY-NC
- **Support**: Questions via [GitHub repository](https://github.com/pytorch/fairseq/tree/master/examples/mms)

## Technical Details

### Model Variants
- **Pre-trained**: 300M and 1B parameter versions
- **Fine-tuned**: 
  - 1B variant for speech recognition
  - 1B variant for language identification

### Training Data
Pre-training datasets:
- VoxPopuli (parliamentary speech)
- MLS (read audiobooks)
- VoxLingua-107 (YouTube speech)
- CommonVoice (read Wikipedia text)
- BABEL (telephone conversations)
- MMS-lab-U (New Testament readings)
- MMS-unlab (various read Christian texts)

Fine-tuning datasets:
- FLEURS
- VoxLingua-107
- MLS
- CommonVoice
- MMS-lab

### Evaluation Metrics
- Character error rate
- Word error rate
- Classification accuracy

## Implementation Considerations

### Use Cases
- **Primary**: Speech processing research across multiple languages
- **Tasks**: Automatic speech recognition, language identification, speech synthesis
- **Warning**: Fine-tuning on other datasets requires additional risk evaluation

### Bias Assessment
- Studies conducted on gender bias and religious language
- Models perform equally well across genders
- Minimal bias detected for religious language (see paper section 8)

## Citation
```
@article{pratap2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
  journal={arXiv},
  year={2023}
}
```

## Contact
- vineelkpratap@meta.com
- androstj@meta.com
- bshi@meta.com
- michaelauli@gmail.com