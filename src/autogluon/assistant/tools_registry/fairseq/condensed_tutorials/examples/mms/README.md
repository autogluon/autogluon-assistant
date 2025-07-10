# Condensed: MMS: Scaling Speech Technology to 1000+ languages

Summary: This tutorial introduces the Massively Multilingual Speech (MMS) project, which provides pre-trained models for speech recognition (ASR), text-to-speech (TTS), and language identification (LID) across 1000+ languages. It demonstrates how to download models from Hugging Face Hub or direct links, and provides code examples for inference with each model type. The tutorial covers command-line usage for transcription, speech synthesis, and language detection, with options for advanced configurations like language model integration. It also references fine-tuning capabilities through Hugging Face Transformers and includes information about GPU-based forced alignment for long audio files.

*This is a condensed version that preserves essential implementation details and context.*

# MMS: Scaling Speech Technology to 1000+ Languages

## Overview
The Massively Multilingual Speech (MMS) project expands speech technology from ~100 languages to over 1,000 languages, with models for speech recognition, language identification, and text-to-speech.

## Available Models

### ASR Models
| Model | Languages | Download | Hub Link |
|---|---|---|---|
| MMS-1B:FL102 | 102 | [model](https://dl.fbaipublicfiles.com/mms/asr/mms1b_fl102.pt) | [🤗 Hub](https://huggingface.co/facebook/mms-1b-fl102) |
| MMS-1B:L1107 | 1107 | [model](https://dl.fbaipublicfiles.com/mms/asr/mms1b_l1107.pt) | [🤗 Hub](https://huggingface.co/facebook/mms-1b-l1107) |
| MMS-1B-all | 1162 | [model](https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt) | [🤗 Hub](https://huggingface.co/facebook/mms-1b-all) |

### TTS Models
Download language-specific models:
```bash
# Download generator only (sufficient for inference)
wget https://dl.fbaipublicfiles.com/mms/tts/eng.tar.gz # English
wget https://dl.fbaipublicfiles.com/mms/tts/azj-script_latin.tar.gz # North Azerbaijani

# Full checkpoint (generator + discriminator + optimizer)
wget https://dl.fbaipublicfiles.com/mms/tts/full_model/eng.tar.gz
```

### LID Models
Models available for 126, 256, 512, 1024, 2048, and 4017 languages on the [🤗 Hub](https://huggingface.co/facebook/mms-lid-126).

## Inference Commands

### ASR Inference
Simple transcription:
```bash
python examples/mms/asr/infer/mms_infer.py --model "/path/to/asr/model" --lang lang_code \
  --audio "/path/to/audio_1.wav" "/path/to/audio_2.wav" "/path/to/audio_3.wav"
```

Advanced configuration with CER/WER calculation:
```bash
PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py \
  -m --config-dir examples/mms/config/ --config-name infer_common \
  decoding.type=viterbi dataset.max_tokens=4000000 \
  distributed_training.distributed_world_size=1 \
  "common_eval.path='/path/to/asr/model'" \
  task.data='/path/to/manifest' dataset.gen_subset="${lang_code}:dev" \
  common_eval.post_process=letter
```

For language model decoding:
```bash
PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py \
  --config-dir=examples/mms/asr/config --config-name=infer_common \
  decoding.type=kenlm distributed_training.distributed_world_size=1 \
  decoding.unique_wer_file=true decoding.beam=500 decoding.beamsizetoken=50 \
  task.data=<MANIFEST_FOLDER_PATH> common_eval.path='<MODEL_PATH.pt>' \
  decoding.lexicon=<LEXICON_FILE> decoding.lmpath=<LM_FILE> \
  decoding.results_path=<OUTPUT_DIR> dataset.gen_subset=${LANG}:dev \
  decoding.lmweight=??? decoding.wordscore=???
```

### TTS Inference
```bash
# English TTS
PYTHONPATH=$PYTHONPATH:/path/to/vits python examples/mms/tts/infer.py \
  --model-dir /path/to/model/eng --wav ./example.wav \
  --txt "Expanding the language coverage of speech technology has the potential to improve access to information for many more people"

# Maithili TTS
PYTHONPATH=$PYTHONPATH:/path/to/vits python examples/mms/tts/infer.py \
  --model-dir /path/to/model/mai --wav ./example.wav \
  --txt "मुदा आइ धरि ई तकनीक सौ सं किछु बेसी भाषा तक सीमित छल जे सात हजार सं बेसी ज्ञात भाषाक एकटा अंश अछी"
```

### LID Inference
```bash
PYTHONPATH='.' python3 examples/mms/lid/infer.py /path/to/dict/l126/ \
  --path /path/to/models/mms1b_l126.pt --task audio_classification \
  --infer-manifest /path/to/manifest.tsv --output-path <OUTDIR>
```

## Fine-tuning

### ASR Fine-tuning
MMS Adapter fine-tuning is available in 🤗 Transformers examples. See the [blog post](https://huggingface.co/blog/mms_adapters) for step-by-step instructions.

### TTS Fine-tuning
Use [this repository](https://github.com/ylacombe/finetune-hf-vits) for fine-tuning MMS TTS checkpoints with 🤗 Transformers.

## Pretrained Models
| Model | Download | Hub |
|---|---|---|
| MMS-300M | [download](https://dl.fbaipublicfiles.com/mms/pretraining/base_300m.pt) | [🤗 Hub](https://huggingface.co/facebook/mms-300m) |
| MMS-1B | [download](https://dl.fbaipublicfiles.com/mms/pretraining/base_1b.pt) | [🤗 Hub](https://huggingface.co/facebook/mms-1b) |

## Forced Alignment
An efficient GPU-based forced alignment algorithm is available for processing long audio files, along with a multilingual alignment model trained on 31K hours of data in 1,130 languages.

## License
CC-BY-NC 4.0