# Condensed: ---

Summary: This tutorial demonstrates how to fine-tune Facebook's Wav2Vec2-BERT model for low-resource ASR using Mongolian as an example. It covers implementing a complete ASR pipeline with Transformers: data preprocessing (audio resampling, text normalization), creating custom tokenizers for non-Latin scripts, configuring CTC-based models with adapters, and training optimization techniques. Key features include efficient processing of speech data with SeamlessM4TFeatureExtractor, specialized data collation for CTC training, performance evaluation using WER metrics, and best practices for low-resource languages (vocabulary cleaning, adapter layers, and optimizer tuning). The tutorial shows how to achieve performance comparable to Whisper models while being 10-30x faster with only 14 hours of training data.

*This is a condensed version that preserves essential implementation details and context.*

# Fine-Tune W2V2-Bert for Low-Resource ASR with 🤗 Transformers

## Introduction

Wav2Vec2-BERT is a 580M-parameter audio model pre-trained on 4.5M hours of unlabeled audio covering 143+ languages. It builds on previous models like Wav2Vec2, XLSR, XLS-R, and MMS, but with significantly more training data. For ASR tasks, it can be fine-tuned using Connectionist Temporal Classification (CTC).

This tutorial demonstrates how to fine-tune the pre-trained checkpoint [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) on a low-resource language (Mongolian) from Common Voice 16.0 with only ~14 hours of validated training data.

## Motivation

While Whisper models provide state-of-the-art ASR performance for many languages, they perform poorly on "resource-poor" languages like Mongolian (100%+ WER in the original paper). Additionally, Whisper:
- Has limited vocabulary for fine-tuning on languages with different alphabets
- Is slow due to its autoregressive nature
- Requires more tokens per word for uncommon languages

Wav2Vec2-BERT offers advantages for low-resource scenarios:
- Predicts ASR in a single pass (much faster)
- Requires little data to achieve competitive performance
- Is easily adaptable to any alphabet
- Is more resource-efficient (2.5x)
- Achieves similar WER to Whisper-large-v3 while being 10-30x faster

## Setup

```bash
pip install datasets transformers torchaudio jiwer accelerate -U
```

For tracking and saving your model:

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Prepare Data, Tokenizer, Feature Extractor

ASR models need:
1. A feature extractor to process speech signals into feature vectors
2. A tokenizer to process model outputs into text

Wav2Vec2-BERT uses:
- [Wav2Vec2CTCTokenizer](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#wav2vec2ctctokenizer)
- [SeamlessM4TFeatureExtractor](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor)

# Create Wav2Vec2CTCTokenizer

## Dataset Preparation

```python
from datasets import load_dataset, load_metric, Audio

# Load Mongolian Common Voice dataset
common_voice_train = load_dataset("mozilla-foundation/common_voice_16_0", "mn", split="train+validation", use_auth_token=True)
common_voice_test = load_dataset("mozilla-foundation/common_voice_16_0", "mn", split="test", use_auth_token=True)

# Remove unnecessary columns
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
```

## Text Normalization

```python
import re
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\"\%\'\"\�\'\»\«]'

def remove_special_characters(batch):
    # Remove special characters and convert to lowercase
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)
```

## Vocabulary Creation

```python
def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

# Extract unique characters from both datasets
vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

# Create vocabulary dictionary
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
```

## Cleaning the Vocabulary

The vocabulary contains a mix of Latin and Mongolian Cyrillic characters. For better CTC performance:

1. Remove Latin characters to reduce vocabulary size
2. Focus only on the Mongolian alphabet

```python
def remove_latin_characters(batch):
    batch["sentence"] = re.sub(r'[a-z]+', '', batch["sentence"])
    return batch
```

**Note**: CTC benefits from a reduced vocabulary size, and removing redundant characters improves performance.

# Data Preprocessing for Mongolian ASR

## Cleaning and Vocabulary Creation

```python
# Remove Latin characters from the dataset
common_voice_train = common_voice_train.map(remove_latin_characters)
common_voice_test = common_voice_test.map(remove_latin_characters)

# Extract unique characters to build vocabulary
vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, 
                                    keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, 
                                  keep_in_memory=True, remove_columns=common_voice_test.column_names)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

# Create vocabulary dictionary
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
```

The resulting vocabulary contains all letters of the Mongolian alphabet.

## Vocabulary Refinement

```python
# Replace space with a visible word delimiter
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add special tokens
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)  # CTC blank token
```

**Note:** Pre-processing is critical for ASR. For example, normalizing case helps the model focus on sounds rather than grammatical rules.

## Saving and Loading the Tokenizer

```python
# Save vocabulary to file
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Create tokenizer from vocabulary
from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", 
                                               unk_token="[UNK]", 
                                               pad_token="[PAD]", 
                                               word_delimiter_token="|")

# Upload tokenizer to Hugging Face Hub
repo_name = "w2v-bert-2.0-mongolian-colab-CV16.0"
tokenizer.push_to_hub(repo_name)
```

## Feature Extractor Setup

```python
# Load feature extractor from pre-trained checkpoint
from transformers import SeamlessM4TFeatureExtractor
feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# Combine feature extractor and tokenizer into a processor
from transformers import Wav2Vec2BertProcessor
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.push_to_hub(repo_name)
```

The feature extractor converts raw audio to log-mel spectrograms that the model can process, while the tokenizer handles text processing. The combined processor simplifies usage by providing a single interface for both audio and text processing.

# Data Preprocessing for Wav2Vec2-BERT

## Audio Data Preparation

The dataset contains three main columns: `sentence` (transcription), `path` (audio file location), and `audio` (audio data). Wav2Vec2-BERT requires input as a 1-dimensional array sampled at 16 kHz.

```python
# Check the audio path
common_voice_train[0]["path"]
# Output: '/root/.cache/huggingface/datasets/downloads/extracted/276aa682ce2b6a24934bc401b1f30e004c3fb178dd41d6295b273329f592844a/mn_train_0/common_voice_mn_18578097.mp3'

# The audio column automatically loads the file
common_voice_train[0]["audio"]
# Shows array data with sampling_rate: 48000
```

### Resampling to 16 kHz

The model requires 16 kHz audio, but our data is 48 kHz. We need to resample:

```python
# Resample audio to 16 kHz
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
```

### Verifying Audio Data

```python
# Verify audio data
rand_int = random.randint(0, len(common_voice_train)-1)
print("Target text:", common_voice_train[rand_int]["sentence"])
print("Input array shape:", common_voice_train[rand_int]["audio"]["array"].shape)
print("Sampling rate:", common_voice_train[rand_int]["audio"]["sampling_rate"])

# Output:
# Target text: энэ бол тэдний амжилтын бодит нууц
# Input array shape: (74496,)
# Sampling rate: 16000
```

## Processing Data for Training

We need to process the data into the format expected by `Wav2Vec2BertForCTC`:

```python
def prepare_dataset(batch):
    audio = batch["audio"]
    # Extract input features using processor (Log-Mel feature extraction)
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])
    
    # Encode transcriptions to label ids
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

# Apply processing to all examples
common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
```

## Training Setup Requirements

For training with the 🤗 Transformers Trainer, we need to:

1. **Define a data collator**: Wav2Vec2-BERT requires a special padding data collator due to the large difference between input and output lengths. Dynamic padding is more efficient.

2. **Create an evaluation metric**: The model should be evaluated on word error rate (WER).

3. **Load a pre-trained checkpoint**: Configure a pre-trained model for fine-tuning.

4. **Define training configuration**: Set up hyperparameters and training settings.

> **Note**: The datasets library automatically handles audio loading and resampling. For custom loading, you can use the "path" column instead of the "audio" column.

# Setting Up the Trainer for Wav2Vec2-BERT

## Data Collator Implementation

```python
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
```

This specialized data collator handles speech input and text output differently, applying separate padding functions to each modality.

## Evaluation Metric

```python
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # Don't group tokens when computing metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

Word Error Rate (WER) is used as the primary evaluation metric for ASR tasks.

## Model Configuration

```python
model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/w2v-bert-2.0",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    add_adapter=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
```

Key configuration details:
- Dropout layers are disabled to prevent overfitting
- Gradient checkpointing is enabled to save GPU memory
- CTC loss reduction is set to "mean"
- An adapter is added for efficient fine-tuning

## Training Arguments

```python
training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=10,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=600,
  eval_steps=300,
  logging_steps=300,
  learning_rate=5e-5,
  warmup_steps=500,
  save_total_limit=2,
  push_to_hub=True,
)
```

Important training configurations:
- `group_by_length=True` improves efficiency by batching samples of similar length
- Learning rate of 5e-5 was heuristically tuned for stability
- FP16 precision is used for faster training
- Checkpoints are saved every 600 steps and uploaded to the Hub
- Gradient accumulation is set to 2 steps

## Trainer Initialization

```python
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)
```

## Important Notes

1. In CTC decoding, consecutive identical tokens are grouped as a single token, but encoded labels should not be grouped when decoding (`group_tokens=False`)

2. The pad token serves as CTC's blank token, allowing the model to predict repeated characters by inserting blank tokens between them

# Training and Evaluation of Wav2Vec2-BERT for Mongolian ASR

## Training

Training the model takes several hours depending on GPU resources. While the results are satisfactory on Common Voice's Mongolian test data, this is not an optimally fine-tuned model.

```python
trainer.train()
```

| Step | Training Loss | Validation Loss | WER      |
|:----:|:------------:|:---------------:|:--------:|
| 300  | 1.712700     | 0.647740        | 0.517892 |
| 600  | 0.349300     | 0.615849        | 0.442027 |
| 900  | 0.180500     | 0.525088        | 0.367305 |
| 1200 | 0.075400     | 0.528768        | 0.324016 |

The training shows good progress with decreasing loss and WER. Notably, this performance is comparable to OpenAI's whisper-large-v3 model, which achieved a final WER of 33.3% on the same dataset. This demonstrates that **Wav2Vec2-BERT can achieve performance equivalent to state-of-the-art models for low-resource languages**.

## Sharing the Model

Upload the trained model to the Hugging Face Hub:

```python
trainer.push_to_hub()
```

Others can then use your model:

```python
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor

model = AutoModelForCTC.from_pretrained("your-username/model-name")
processor = Wav2Vec2BertProcessor.from_pretrained("your-username/model-name")
```

## Evaluation

Let's verify the model's performance on Mongolian speech:

```python
model = Wav2Vec2BertForCTC.from_pretrained(repo_name).to("cuda")
processor = Wav2Vec2BertProcessor.from_pretrained(repo_name)

sample = common_voice_test[0]
input_features = torch.tensor(sample["input_features"]).to("cuda").unsqueeze(0)

with torch.no_grad():
    logits = model(input_features).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

print(processor.decode(pred_ids))
print(processor.decode(sample["labels"]).lower())
```

Output:
```
эрчүүдийн ганцаардлыг эмэхтэйчүүд ойлгох нь ховор юм
эрчүдийн ганцардлыг эмэгтэйчүд ойлгох нь ховор юм
```

The transcription is recognizable but not perfect. Performance could be improved with:
- Longer training
- Better data pre-processing
- Using a language model for decoding

## Scaling-up Training: Best Practices

### Dataset-Related Tips

1. **Use lowercase, unpunctuated transcriptions** for CTC ASR, as the model should focus on acoustic prediction rather than language modeling.

2. **Remove low-frequency characters** from the tokenizer vocabulary:
   - Very low-frequency characters can cause loss spikes during training
   - Characters that appear rarely should be treated as errors and classified as `"[UNK]"`
   - Common Voice datasets often contain "wrong" characters from other languages

3. **Use the newest Common Voice version** (CV16) which provides more hours of data for many languages, enabling more efficient models for low-resource languages.

# Training-related Tips for W2V-BERT

## Optimal CTC Token Duration

The ideal ratio of duration seen per CTC token should be **10 to 35 ms**. This corresponds to a fraction of the time needed to pronounce a phoneme.

**Problem Example:** In one training run, the loss curve initially decreased but later exploded because each CTC token was seeing 30-60 ms of signal (too long).

**Solution:** Add a convolutional adapter layer to sub-sample the encoder hidden states:

```python
# Configure adapter in Wav2Vec2BertConfig
config = Wav2Vec2BertConfig.from_pretrained(
    "facebook/w2v-bert-2.0",
    add_adapter=True  # Add this parameter to reduce time dimension
)
```

## Addressing Under-training Issues

Signs of under-training:
- Loss curve stops during steep descent
- Lack of smoothness in the loss curve

**Solutions:**
1. **Adjust warm-up rate:**
   - Keep warm-up ratio between 5-15%
   - Scale up the number of epochs
   - Warm-up steps help align new language-model head weights with the pre-trained model

2. **Tune optimizer parameters:**
   - Adjust AdamW's β₂ parameter (typically 0.95-0.98) to improve loss curve smoothness

```python
# Example of optimizer configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.98)  # Adjusted β₂ for smoother loss curve
)
```

## Related Resources
- [Official paper](https://huggingface.co/papers/2305.13516)
- [Original codebase](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
- [Transformers Docs](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2-bert)
- [XLS-R blog post](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)
- [MMS blog post](https://huggingface.co/blog/mms_adapters)