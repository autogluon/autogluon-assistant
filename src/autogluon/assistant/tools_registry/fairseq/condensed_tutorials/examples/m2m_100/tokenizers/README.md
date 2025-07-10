# Condensed: M2M-100 Tokenization

Summary: This tutorial explains M2M-100's language-specific tokenization implementation for accurate multilingual machine translation evaluation. It demonstrates how to use the tok.sh script to properly tokenize generated outputs and reference translations before calculating BLEU scores with sacrebleu. The tutorial covers the reproduction workflow with command-line instructions, installation requirements for tokenization dependencies, and special handling for Arabic evaluation. This knowledge helps with implementing proper tokenization for multilingual NLP tasks, evaluating machine translation quality, and reproducing M2M-100 results accurately.

*This is a condensed version that preserves essential implementation details and context.*

# M2M-100 Tokenization

## Implementation Overview

M2M-100 applies language-specific tokenization strategies to reproduce results accurately. The provided `tok.sh` script handles tokenization for different languages.

## Reproduction Steps

```bash
tgt_lang=...
reference_translation=...
cat generation_output | grep -P "^H" | sort -V | cut -f 3- | sh tok.sh $tgt_lang > hyp
cat $reference_translation | sh tok.sh $tgt_lang > ref
sacrebleu -tok 'none' ref < hyp
```

## Installation Requirements

- Run `install_dependencies.sh` to install tools for all languages except Arabic
- For Arabic evaluation, follow installation instructions at: http://alt.qcri.org/tools/arabic-normalizer/