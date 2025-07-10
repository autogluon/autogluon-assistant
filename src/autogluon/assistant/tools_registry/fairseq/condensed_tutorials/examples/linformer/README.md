# Condensed: Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)

Summary: This tutorial provides implementation guidance for Linformer, a transformer architecture achieving linear complexity self-attention. It demonstrates how to train a Linformer RoBERTa model by modifying standard RoBERTa pretraining with specific command-line arguments. The tutorial helps with implementing efficient transformer models for NLP tasks where standard quadratic-complexity attention is prohibitive. Key features include the linear complexity self-attention mechanism based on Wang et al.'s 2020 paper, which enables processing longer sequences more efficiently than traditional transformer architectures.

*This is a condensed version that preserves essential implementation details and context.*

# Linformer: Self-Attention with Linear Complexity

## Implementation Details

To train a Linformer RoBERTa model, follow the standard RoBERTa pretraining process with these key modifications:

```bash
--user-dir examples/linformer/linformer_src --arch linformer_roberta_base
```

This implementation allows for self-attention with linear complexity as described in [Wang et al.'s paper (2020)](https://arxiv.org/abs/2006.04768).

## Citation

```bibtex
@article{wang2020linformer,
  title={Linformer: Self-Attention with Linear Complexity},
  author={Wang, Sinong and Li, Belinda and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
```