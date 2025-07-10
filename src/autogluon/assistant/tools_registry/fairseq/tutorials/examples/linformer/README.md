Summary: This tutorial provides implementation guidance for Linformer, a transformer architecture achieving linear complexity self-attention. It demonstrates how to train a Linformer RoBERTa model by modifying standard RoBERTa pretraining with specific command-line arguments. The tutorial helps with implementing efficient transformer models for NLP tasks where standard quadratic-complexity attention is prohibitive. Key features include the linear complexity self-attention mechanism based on Wang et al.'s 2020 paper, which enables processing longer sequences more efficiently than traditional transformer architectures.

# Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)

This example contains code to train Linformer models as described in our paper
[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768).

## Training a new Linformer RoBERTa model

You can mostly follow the [RoBERTa pretraining README](/examples/roberta/README.pretraining.md),
updating your training command with `--user-dir examples/linformer/linformer_src --arch linformer_roberta_base`.

## Citation

If you use our work, please cite:

```bibtex
@article{wang2020linformer,
  title={Linformer: Self-Attention with Linear Complexity},
  author={Wang, Sinong and Li, Belinda and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
```
