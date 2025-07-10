# Condensed: Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)

Summary: This tutorial explains LayerDrop, a structured dropout technique for training deeper transformers and pruning them at inference time. It covers implementation details including encoder/decoder layerdrop parameters and layer selection for pruning. The tutorial provides code examples for training with LayerDrop, pruning models, evaluation with pruned models, and using pre-trained RoBERTa+LayerDrop. It offers best practices for performance optimization, aggressive pruning strategies, layer distribution, model compatibility, and regularization balance. The tutorial also lists available pre-trained models with LayerDrop for translation and language understanding tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Reducing Transformer Depth with Structured Dropout (LayerDrop)

## Implementation Details

LayerDrop is a structured dropout technique that enables training deeper models and pruning them at inference time.

### Key Parameters
- `--encoder-layerdrop 0.2` - Dropout rate for encoder layers
- `--decoder-layerdrop 0.2` - Dropout rate for decoder layers
- `--encoder-layers-to-keep 0,2,4,6,8,10,12,14` - Specify which layers to keep during pruning
- `--decoder-layers-to-keep 0,2,4,6,8,10,12,14` - Specify which layers to keep during pruning

## Code Examples

### Training with LayerDrop
```bash
# Add these flags to your training command
--encoder-layerdrop 0.2 --decoder-layerdrop 0.2
```

### Pruning a LayerDrop Model
```bash
# Add these flags to keep specific layers
--encoder-layers-to-keep 0,2,4,6,8,10,12,14 --decoder-layers-to-keep 0,2,4,6,8,10,12,14
```

### Evaluation with a Pruned Model
```bash
fairseq-eval-lm /path/to/wikitext-103 \
  --path /path/to/model/checkpoint.pt \
  --model-overrides "{'decoder_layers_to_keep':'0,2,4,6,8,10,12,14'}"
```

### Using Pre-trained RoBERTa+LayerDrop
```python
from fairseq.models.roberta import RobertaModel

roberta_layerdrop = RobertaModel.from_pretrained(
    '/path/to/MNLI/model',
    checkpoint_file='mnli_checkpoint.pt',
    data_name_or_path='/path/to/MNLI/data/MNLI-bin'
)

# Evaluation code
roberta_layerdrop.cuda()
roberta_layerdrop.eval()
# Process data and make predictions
```

## Best Practices

1. **For better performance**: Use smaller LayerDrop values (0.1-0.2) and slightly reduce standard dropout.

2. **For aggressive pruning**: Use larger LayerDrop values (~0.5) if you plan to remove half or more of the network.

3. **When pruning layers**: Distribute remaining layers evenly throughout the network (e.g., keep every other layer when removing 50%).

4. **Model compatibility**: Training from scratch with LayerDrop works better than adding it during fine-tuning for pre-trained models like BERT/RoBERTa.

5. **Regularization balance**: If your model is underfitting, LayerDrop may add too much regularization - use smaller values or reduce standard dropout.

## Available Pre-trained Models
- `layerdrop_wmt_en_de_12_6`: Transformer + LayerDrop 0.2 for translation
- `roberta_layerdrop.base`: RoBERTa Base + LayerDrop 0.2
- `roberta_layerdrop.large`: RoBERTa Large + LayerDrop 0.2
- Fine-tuned versions for MNLI and QNLI tasks