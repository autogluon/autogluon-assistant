Summary: This tutorial explains M2M-100's language-specific tokenization implementation for accurate multilingual machine translation evaluation. It demonstrates how to use the tok.sh script to properly tokenize generated outputs and reference translations before calculating BLEU scores with sacrebleu. The tutorial covers the reproduction workflow with command-line instructions, installation requirements for tokenization dependencies, and special handling for Arabic evaluation. This knowledge helps with implementing proper tokenization for multilingual NLP tasks, evaluating machine translation quality, and reproducing M2M-100 results accurately.

# M2M-100 Tokenization

We apply different tokenization strategies for different languages following the existing literature. Here we provide tok.sh a tokenizer that can be used to reproduce our results.

To reproduce the results, follow these steps:

```
tgt_lang=...
reference_translation=...
cat generation_output | grep -P "^H" | sort -V | cut -f 3- | sh tok.sh $tgt_lang > hyp
cat $reference_translation |sh tok.sh $tgt_lang > ref
sacrebleu -tok 'none' ref < hyp
```

## Installation

Tools needed for all the languages except Arabic can be installed by running install_dependencies.sh
If you want to evaluate Arabic models, please follow the instructions provided here: http://alt.qcri.org/tools/arabic-normalizer/ to install 
