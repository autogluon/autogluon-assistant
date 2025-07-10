# Condensed: Discriminative Reranking for Neural Machine Translation

Summary: This tutorial implements Discriminative Reranking for Neural Machine Translation (DrNMT), a technique to improve translation quality by reranking candidate translations. It covers: (1) data preparation including processing source/target sentences and generating hypothesis candidates, (2) training a reranker model using XLM-RoBERTa, and (3) inference with weight tuning. Key functionalities include integrating with fairseq, processing multiple hypotheses per source sentence, optimizing for BLEU/TER metrics, and applying trained rerankers to test data. The code helps with building advanced translation systems that leverage pretrained language models to select better translations from n-best lists.

*This is a condensed version that preserves essential implementation details and context.*

# Discriminative Reranking for Neural Machine Translation (DrNMT)

## Data Preparation

1. **Build base MT model** and prepare three files:
   - Source sentences file (L lines)
   - Target sentences file (L lines)
   - Hypotheses file (L*N lines, where N is number of hypotheses per source)

2. **Download XLMR model**:
   ```bash
   wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz
   tar zxvf xlmr.base.tar.gz
   ```

3. **Prepare scores and BPE data**:
   ```bash
   python scripts/prep_data.py \
       --input-source ${SOURCE_FILE} \
       --input-target ${TARGET_FILE} \
       --input-hypo ${HYPO_FILE} \
       --output-dir ${OUTPUT_DIR} \
       --split $SPLIT \
       --beam $N \
       --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
       --metric $METRIC \
       --num-shards ${NUM_SHARDS}
   ```
   - `N`: Number of hypotheses per source (50 in paper)
   - `METRIC`: Either `bleu` or `ter`
   - `NUM_SHARDS`: Use 1 for non-train splits

4. **Pre-process into fairseq format**:
   ```bash
   # Process first shard
   for suffix in src tgt ; do
       fairseq-preprocess --only-source \
           --trainpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/train.bpe \
           --validpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid.bpe \
           --destdir ${OUTPUT_DIR}/$METRIC/split1/input_${suffix} \
           --workers 60 \
           --srcdict ${XLMR_DIR}/dict.txt
   done
   
   # Process additional shards
   for i in `seq 2 ${NUM_SHARDS}`; do
       # [Additional shard processing code]
   done
   ```

## Training

```bash
fairseq-hydra-train -m \
    --config-dir config/ --config-name deen \
    task.data=${OUTPUT_DIR}/$METRIC/split1/ \
    task.num_data_splits=${NUM_SHARDS} \
    model.pretrained_model=${XLMR_DIR}/model.pt \
    common.user_dir=${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
    checkpoint.save_dir=${EXP_DIR}
```

**Important configurations**:
- For fewer GPUs: Set `distributed_training.distributed_world_size=k +optimization.update_freq='[x]'` where x = 16/k
- For fewer hypotheses: Set `task.mt_beam=N dataset.batch_size=N dataset.required_batch_size_multiple=N`

## Inference & Scoring

1. **Tune weights on validation set**:
   ```bash
   # Generate N hypotheses with base MT model
   cat ${VALID_SOURCE_FILE} | \
       fairseq-interactive ${MT_DATA_PATH} \
       --max-tokens 4000 --buffer-size 16 \
       --num-workers 32 --path ${MT_MODEL} \
       --beam $N --nbest $N \
       --post-process sentencepiece &> valid-hypo.out
   
   # Tune weights
   python drnmt_rerank.py \
       ${OUTPUT_DIR}/$METRIC/split1/ \
       --path ${EXP_DIR}/checkpoint_best.pt \
       --in-text valid-hypo.out \
       --results-path ${EXP_DIR} \
       --gen-subset valid \
       --target-text ${VALID_TARGET_FILE} \
       --user-dir ${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
       --bpe sentencepiece \
       --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
       --beam $N \
       --batch-size $N \
       --metric bleu \
       --tune
   ```

2. **Apply best weights on test sets**:
   ```bash
   # Generate hypotheses
   cat ${TEST_SOURCE_FILE} | \
       fairseq-interactive ${MT_DATA_PATH} \
       --max-tokens 4000 --buffer-size 16 \
       --num-workers 32 --path ${MT_MODEL} \
       --beam $N --nbest $N \
       --post-process sentencepiece &> test-hypo.out
   
   # Apply reranking
   python drnmt_rerank.py \
       ${OUTPUT_DIR}/$METRIC/split1/ \
       --path ${EXP_DIR}/checkpoint_best.pt \
       --in-text test-hypo.out \
       --results-path ${EXP_DIR} \
       --gen-subset test \
       --user-dir ${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
       --bpe sentencepiece \
       --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
       --beam $N \
       --batch-size $N \
       --metric bleu \
       --fw-weight ${BEST_FW_WEIGHT} \
       --lenpen ${BEST_LENPEN}
   ```
   - Add `--target-text` to evaluate BLEU/TER, otherwise only generates highest-scoring hypotheses