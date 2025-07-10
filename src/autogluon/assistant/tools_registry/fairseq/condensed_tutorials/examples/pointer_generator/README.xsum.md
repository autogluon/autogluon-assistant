# Condensed: Training a pointer-generator model on the Extreme Summarization dataset

Summary: This tutorial demonstrates implementing a Pointer-Generator model for extreme text summarization using Fairseq. It covers preprocessing techniques for handling unknown words with position markers, model configuration with transformer architecture, and inference procedures. Key features include vocabulary creation with source position markers, transformer-based pointer-generator architecture configuration, and post-processing to replace unknown tokens with original words. The tutorial provides complete code for data preparation, model training with specific hyperparameters, and inference generation, making it valuable for implementing advanced text summarization systems.

*This is a condensed version that preserves essential implementation details and context.*

# Training a Pointer-Generator Model on Extreme Summarization Dataset

## Data Preparation

1. **Download and preprocess the dataset**
   - Obtain the Extreme Summarization dataset from [Edinburgh NLP](https://github.com/EdinburghNLP/XSum)

2. **Create vocabulary with source position markers**
   ```bash
   vocab_size=10000
   position_markers=1000
   cat train.document train.summary |
     tr -s '[:space:]' '\n' |
     sort | uniq -c | sort -k1,1bnr -k2 |
     head -n "$((vocab_size - 4))" |
     awk '{ print $2 " " $1 }' >dict.pg.txt
   python3 -c "[print('<unk-{}> 0'.format(n)) for n in range($position_markers)]" >>dict.pg.txt
   ```

3. **Preprocess text data**
   ```bash
   ./preprocess.py --source train.document --target train.summary --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out train.pg.src --target-out train.pg.tgt
   ./preprocess.py --source validation.document --target validation.summary --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out valid.pg.src --target-out valid.pg.tgt
   ./preprocess.py --source test.document --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out test.pg.src
   ```

4. **Binarize dataset**
   ```bash
   fairseq-preprocess \
     --source-lang src \
     --target-lang tgt \
     --trainpref train.pg \
     --validpref valid.pg \
     --destdir bin \
     --workers 60 \
     --srcdict dict.pg.txt \
     --joined-dictionary
   ```

## Model Training

```bash
total_updates=20000
warmup_updates=500
lr=0.001
max_tokens=4096
update_freq=4
pointer_layer=-2

fairseq-train bin \
    --user-dir examples/pointer_generator/pointer_generator_src \
    --max-tokens "$max_tokens" \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --required-batch-size-multiple 1 \
    --arch transformer_pointer_generator \
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --source-position-markers 1000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr "$lr" \
    --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --update-freq "$update_freq" \
    --skip-invalid-size-inputs-valid-test
```

**Key configuration details:**
- Uses 1000 source position markers
- Uses one attention head from penultimate decoder layer for pointing
- Training takes ~5.5 hours on eight 32GB V100 GPUs
- Dictionary indices above 10000 map to `<unk>` embedding

## Inference and Post-processing

1. **Generate summaries**
   ```bash
   batch_size=32
   beam_size=6
   max_length=60
   length_penalty=1.0

   fairseq-interactive bin \
       --user-dir examples/pointer_generator/pointer_generator_src \
       --batch-size "$batch_size" \
       --task translation \
       --source-lang src --target-lang tgt \
       --path checkpoints/checkpoint_last.pt \
       --input test.pg.src \
       --buffer-size 200 \
       --max-len-a 0 \
       --max-len-b "$max_length" \
       --lenpen "$length_penalty" \
       --beam "$beam_size" \
       --skip-invalid-size-inputs-valid-test |
       tee generate.out
   grep ^H generate.out | cut -f 3- >generate.hyp
   ```

2. **Post-process output** (replace `<unk-N>` tokens with original words)
   ```bash
   ./postprocess.py \
       --source <(awk 'NF<1024' test.document) \
       --target generate.hyp \
       --target-out generate.hyp.processed
   ```

## Example Output

**Original document:**
> de roon moved to teesside in june 2016 for an initial # 8.8 m fee and played 33 premier league games last term. the netherlands international, 26, scored five goals in 36 league and cup games during his spell at boro. meanwhile, manager garry monk confirmed the championship club's interest in signing chelsea midfielder lewis baker. "he's a target and one of many that we've had throughout the summer months," said monk. find all the latest football transfers on our dedicated page.

**Preprocessed source:**
> de \<unk-1> moved to \<unk-4> in june 2016 for an initial # \<unk-12> m fee and played 33 premier league games last term . the netherlands international , 26 , scored five goals in 36 league and cup games during his spell at boro . meanwhile , manager garry monk confirmed the championship club 's interest in signing chelsea midfielder lewis baker . `` he 's a target and one of many that we 've had throughout the summer months , '' said monk . find all the latest football transfers on our dedicated page .

**Generated summary after post-processing:**
> middlesbrough striker \<unk> de roon has joined spanish side \<unk> on a season-long loan.