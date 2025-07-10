# Condensed: Hierarchical Neural Story Generation (Fan et al., 2018)

Summary: This tutorial implements hierarchical neural story generation based on Fan et al.'s 2018 paper. It covers: (1) setting up and preprocessing the WritingPrompts dataset, including trimming stories to 1000 words; (2) training a convolutional self-attention model for story generation using fairseq; (3) implementing fusion models with pretrained checkpoints; and (4) generating creative text with sampling techniques. Key functionalities include data binarization, model training with specific attention mechanisms (gated/self-attention), and text generation with temperature sampling and beam search parameters for controlling story output quality.

*This is a condensed version that preserves essential implementation details and context.*

# Hierarchical Neural Story Generation Implementation

## Pre-trained Models
- **Stories with Convolutional Model** ([Fan et al., 2018](https://arxiv.org/abs/1805.04833))
  - Dataset: [WritingPrompts](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz)
  - Model: [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/stories_checkpoint.tar.bz2)
  - Test set: [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/stories_test.tar.bz2)

## Dataset Setup
```bash
cd examples/stories
curl https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz | tar xvzf -
```

## Data Preprocessing
The paper models only the first 1000 words of each story:

```python
# Trim stories to first 1000 words
data = ["train", "test", "valid"]
for name in data:
    with open(name + ".wp_target") as f:
        stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    with open(name + ".wp_target", "w") as o:
        for line in stories:
            o.write(line.strip() + "\n")
```

## Training Pipeline

### 1. Binarize the dataset
```bash
export TEXT=examples/stories/writingPrompts
fairseq-preprocess --source-lang wp_source --target-lang wp_target \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/writingPrompts --padding-factor 1 \
    --thresholdtgt 10 --thresholdsrc 10
```

### 2. Train the model
```bash
fairseq-train data-bin/writingPrompts -a fconv_self_att_wp --lr 0.25 \
    --optimizer nag --clip-norm 0.1 --max-tokens 1500 \
    --lr-scheduler reduce_lr_on_plateau --decoder-attention True \
    --encoder-attention False --criterion label_smoothed_cross_entropy \
    --weight-decay .0000001 --label-smoothing 0 \
    --source-lang wp_source --target-lang wp_target \
    --gated-attention True --self-attention True \
    --project-input True --pretrained False
```

### 3. Train a fusion model (optional)
Add these arguments to the training command:
```
--pretrained True --pretrained-checkpoint path/to/checkpoint
```

### 4. Generate text
```bash
fairseq-generate data-bin/writingPrompts \
    --path /path/to/trained/model/checkpoint_best.pt --batch-size 32 \
    --beam 1 --sampling --sampling-topk 10 --temperature 0.8 --nbest 1 \
    --model-overrides "{'pretrained_checkpoint':'/path/to/pretrained/model/checkpoint'}"
```

**Important note**: When loading a pretrained model at generation time, use `--model-overrides` to specify the pretrained checkpoint path if you've moved it from its original location.