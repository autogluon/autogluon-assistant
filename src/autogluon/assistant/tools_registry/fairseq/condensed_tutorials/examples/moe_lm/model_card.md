# Condensed: Model card for the paper ``Efficient Large Scale Language Modeling with Mixtures of Experts"

Summary: This tutorial covers Mixture of Experts (MoE) architecture implementation for efficient large language models, demonstrating how to build models with up to 1.1T parameters while maintaining computational efficiency. It helps with coding tasks related to sparse computation in neural networks, where only a subset of parameters activate for each input. Key features include: implementing MoE architectures that achieve 160x parameter increase with only 30% more FLOPS, evaluation methodologies across zero-shot/few-shot/supervised settings, and techniques for measuring model bias and fairness. The tutorial provides practical knowledge for developers seeking to build more efficient large-scale language models.

*This is a condensed version that preserves essential implementation details and context.*

# Model Card: Efficient Large Scale Language Modeling with Mixtures of Experts

## Overview
- **Developer**: FAIR (Fundamental Artificial Intelligence Research)
- **Model Types**:
  - Dense models (125M to 13B parameters)
  - Sparse MoE models (15B to 1.1T parameters)
- **Primary Use**: Research purposes only

## Key Technical Details

### Model Architecture
The 1.1T parameter model uses a Mixture of Experts (MoE) architecture that leverages sparse computation - only a small fraction of parameters are active for any given input. This provides significant efficiency benefits:
- 160x increase in parameters with only 30% increase in FLOPS compared to a 6.7B dense model
- Better validation perplexity for a given compute budget compared to dense models

### Evaluation Methodology

#### Zero-Shot Evaluation
- **HellaSwag**: Commonsense reasoning
- **PIQA**: Physical commonsense reasoning
- **ReCoRD**: Reading comprehension with commonsense reasoning

#### Few-Shot Evaluation
- **Implementation detail**: Average results across 25 runs, randomly sampling different few-shot examples each time
- **Datasets**: Winogrande, StoryCloze, OpenBookQA

#### Fully Supervised Evaluation
- **BoolQ**: Yes/no question answering
- **SST-2**: Binary sentiment classification
- **MNLI**: Natural language inference across multiple genres

### Training Data
Six English language datasets totaling ~450GB:
- BookCorpus (4GB)
- English Wikipedia (12GB)
- CC-News (76GB)
- OpenWebText (38GB)
- CC-Stories (31GB)
- English CC100 (292GB)

## Responsible AI Considerations

### Fairness and Bias
- Both dense and MoE models get worse in terms of Stereotype Score (SS) with scale
- Evaluated on StereoSet and CrowS-pairs datasets to quantify encoded bias in:
  - Gender
  - Profession
  - Race
  - Religion
  - Age

### Efficiency (Green AI)
The MoE architecture provides significant computational efficiency:
- Sparse activation pattern means only a subset of the full parameter count is used for any given input
- 1.1T parameter model requires only 30% more FLOPS than a 6.7B dense model

## Important Notes
- The model is not primarily intended for language generation
- Performance metrics focus on perplexity, zero-shot/few-shot learning, and supervised fine-tuning
- Model exhibits biases that increase with scale, as measured by standard bias evaluation datasets