# Condensed: Finetuning RoBERTa on GLUE tasks

Summary: This tutorial demonstrates how to fine-tune RoBERTa models on GLUE benchmark tasks. It covers the complete implementation workflow including data downloading, preprocessing with task-specific scripts, and model fine-tuning using fairseq-hydra-train with configuration files. The tutorial also provides inference code for evaluating fine-tuned models, showing how to load models, process input pairs, make predictions, and calculate accuracy. Key functionalities include handling various GLUE tasks (QQP, MNLI, QNLI, etc.), GPU-based training configuration, and sentence classification with pre-trained language models.

*This is a condensed version that preserves essential implementation details and context.*

# Finetuning RoBERTa on GLUE Tasks

## Setup Process

1. **Download GLUE data**:
   ```bash
   wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
   python download_glue_data.py --data_dir glue_data --tasks all
   ```

2. **Preprocess data**:
   ```bash
   ./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
   ```
   Where `<glue_task_name>` is one of: `ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA`

3. **Fine-tune model**:
   ```bash
   ROBERTA_PATH=/path/to/roberta/model.pt
   
   CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train -config-dir examples/roberta/config/finetuning \
   --config-name rte task.data=RTE-bin checkpoint.restore_file=$ROBERTA_PATH
   ```

## Important Notes

- Tested on Nvidia V100 GPU (32GB memory)
- Adjust `--update-freq` and `--batch-size` based on available GPU memory
- Task-specific config files are available in `examples/roberta/config/finetuning`

## Inference Code

```python
from fairseq.models.roberta import RobertaModel

# Load fine-tuned model
roberta = RobertaModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='RTE-bin'
)

# Helper function to convert label IDs to strings
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

# Evaluation setup
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

# Run inference on dev set
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1

print('| Accuracy: ', float(ncorrect)/float(nsamples))
```