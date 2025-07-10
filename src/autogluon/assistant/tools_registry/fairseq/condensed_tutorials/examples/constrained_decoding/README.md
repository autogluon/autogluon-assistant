# Condensed: (Vectorized) Lexically constrained decoding with dynamic beam allocation

Summary: This tutorial explains Fairseq's lexically constrained decoding implementation, which forces specific words to appear in generated text. It covers how to use the `--constraints` flag with `fairseq-interactive` to control machine translation output with ordered or unordered constraints. The implementation is based on two academic papers and includes the `LexicallyConstrainedBeamSearch` class and constraint state classes. Key features include efficient beam allocation, support for both ordered and unordered constraints, and effectiveness with smaller beam sizes (5-10). The tutorial provides command-line examples and implementation details useful for integrating constrained decoding into NLP applications.

*This is a condensed version that preserves essential implementation details and context.*

# Lexically Constrained Decoding in Fairseq

This implementation follows the papers:
- [Fast Lexically Constrained Decoding With Dynamic Beam Allocation](https://www.aclweb.org/anthology/N18-1119/) (Post & Vilar, 2018)
- [Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting](https://www.aclweb.org/anthology/N19-1090/) (Hu et al., 2019)

## Usage

Enable constrained search with `--constraints` flag in `fairseq-interactive`. Constraints are added as tab-separated fields after each input line.

```bash
echo -e "Die maschinelle Übersetzung ist schwer zu kontrollieren.\thard\ttoinfluence" \
| normalize.py | tok.py \
| fairseq-interactive /path/to/model \
  --path /path/to/model/model1.pt \
  --bpe fastbpe \
  --bpe-codes /path/to/model/bpecodes \
  --constraints \
  -s de -t en \
  --beam 10
```

By default, constraints are generated in the order provided. For unordered constraints, use `--constraints unordered` (may require larger beam).

## Implementation Details

The implementation is in:
- `fairseq/search.py`: Contains `LexicallyConstrainedBeamSearch` class
- `fairseq/token_generation_contstraints.py`: Contains two constraint state classes:
  - `OrderedConstraintState`: Applies constraints in provided order
  - `UnorderedConstraintState`: Tries all possible orders of constraints (C! permutations)

## Key Advantages

- Ordered constraint generation (default) 
- Improved beam allocation eliminates need for beam pruning
- Effective with smaller beam sizes (5-10 often sufficient)
- Includes vector extensions from Hu et al.

## Best Practices

- For ordered constraints, beam size of 10 is often sufficient
- For unordered constraints, larger beam sizes may be needed
- No beam pruning required due to improved allocation method