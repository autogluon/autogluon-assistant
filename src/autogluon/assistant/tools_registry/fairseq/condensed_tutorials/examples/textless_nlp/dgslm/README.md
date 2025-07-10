# Condensed: Generative Spoken Dialogue Language Modeling

Summary: This tutorial presents the Generative Spoken Dialogue Language Modeling (dGSLM) framework for generating spoken dialogues directly from audio without text. It covers three key components: a Speech-to-Unit Encoder (Fisher HuBERT), a Unit-to-Speech Decoder (HiFi-GAN Vocoder), and a Spoken Dialogue Transformer Language Model (SpeechDLM). The tutorial provides implementation details for loading pre-trained models, sampling from trained models, and training custom SpeechDLM models. It includes code examples for model initialization, input sequence definition, generation, data preprocessing, model training with specific parameters, and validation. This resource is valuable for implementing speech-to-speech dialogue systems without intermediate text representations.

*This is a condensed version that preserves essential implementation details and context.*

# Generative Spoken Dialogue Language Modeling

This repository contains code and pre-trained models for generating naturalistic spoken dialogues directly from audio without text.

## Key Components

### 1. Speech-to-Unit Encoder (Fisher HuBERT model)
Pre-trained models to produce discrete units for the dGSLM model.

### 2. Unit-to-Speech Decoder (HiFi-GAN Vocoder)
Synthesizes waveforms from discrete units.

### 3. Spoken Dialogue Transformer Language Model (SpeechDLM)

#### Pre-trained Model
```
Pre-trained SpeechDLM model (DLM-5 with Edge Unit Prediction & Delayed Duration Prediction):
- model checkpoint: https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/speech_dlm/speech_dlm_base.pt
- dictionary 1: https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/speech_dlm/dict.unitA.txt
- dictionary 2: https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/speech_dlm/dict.unitB.txt
```

#### Sampling from a Trained Model

```python
from fairseq.models.speech_dlm import SpeechDLM

# Load SpeechDLM model
speech_dlm = SpeechDLM.from_pretrained(
                model_name_or_path='/path/to/model/dir',
                checkpoint_file='speech_dlm_base.pt',
                data_name_or_path='/path/to/data/dir'
            )
speech_dlm.eval()
speech_dlm.cuda()

# Define input sequences
input_sequences = [{
      'unitA': '7 376 376 133 178 486 486 486 486 486 486 486 486 2 486',
      'unitB': '7 499 415 177 7 7 7 7 7 7 136 136 289 289 408'
    }]

# Sample from the model
generated_units = speech_dlm.sample(
        input_sequences,
        max_len_a=0,
        max_len_b=500,
        sampling=True,
        beam=5,
    )
```

Or using the script:
```bash
python sample_speech_dlm.py \
    --in-file $INPUT_CODE_FILE --out-file $OUTPUT_FILE \
    --ckpt $CHECKPOINT_PATH --data $DATA_DIR
```

#### Training a SpeechDLM Model

1. **Data Preparation**
   - Prepare two files per split (train, valid) for each channel (unitA, unitB)
   - Preprocess with `fairseq-preprocess`:
   ```bash
   # Preprocess first channel
   fairseq-preprocess --source-lang unitA \
       --only-source \
       --trainpref $RAW_DATA_DIR/train \
       --validpref $RAW_DATA_DIR/valid \
       --destdir $BIN_DATA_DIR \
       --workers 20

   # Preprocess second channel
   fairseq-preprocess --source-lang unitB \
       --srcdict $BIN_DATA_DIR/dict.unitA.txt \
       --only-source \
       --trainpref $RAW_DATA_DIR/train \
       --validpref $RAW_DATA_DIR/valid \
       --destdir $BIN_DATA_DIR \
       --workers 20

   # Rename files
   for channel in unitA unitB; do
     for split in train valid; do
       mv $BIN_DATA_DIR/${split}.${channel}-None.${channel}.bin $BIN_DATA_DIR/${split}.${channel}.bin
       mv $BIN_DATA_DIR/${split}.${channel}-None.${channel}.idx $BIN_DATA_DIR/${split}.${channel}.idx
     done
   done
   ```

2. **Train the Model**
   ```bash
   fairseq-train $BIN_DATA_DIR \
       --save-dir $CHECKPOINT_DIR \
       --tensorboard-logdir $CHECKPOINT_DIR \
       --task speech_dlm_task --channels unitA,unitB \
       --next-unit-prediction "False" --edge-unit-prediction "True" \
       --duration-prediction "True" --delayed-duration-target "True" \
       --criterion speech_dlm_criterion \
       --arch speech_dlm --decoder-cross-layers 4 \
       --share-decoder-input-output-embed \
       --dropout 0.1 --attention-dropout 0.1 \
       --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 \
       --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
       --max-tokens 18432 --tokens-per-sample 6144 --sample-break-mode none \
       --update-freq 16 --num-workers 4 --skip-invalid-size-inputs-valid-test \
       --max-update 250000 --warmup-updates 20000 \
       --save-interval-updates 10000 --keep-last-epochs 1 --no-epoch-checkpoints \
       --log-interval 50 --seed 100501 \
       --fp16 --checkpoint-activations
   ```

3. **Validate**
   ```bash
   fairseq-validate $BIN_DATA_DIR \
       --task speech_dlm_task \
       --path $CHECKPOINT_PATH \
       --max-tokens 6144
   ```

## Citation
```bibtex
@article{nguyen2022dgslm,
  title   = {Generative Spoken Dialogue Language Modeling},
  author  = {Nguyen, Tu Anh and Kharitonov, Eugene and Copet, Jade and Adi, Yossi and Hsu, Wei-Ning and Elkahky, Ali and Tomasello, Paden and Algayres, Robin and Sagot, Benoit and Mohamed, Abdelrahman and Dupoux, Emmanuel},
  eprint={2203.16502},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```