# Condensed: ASR-BLEU evaluation toolkit

Summary: This tutorial presents an ASR-BLEU Evaluation Toolkit for speech-to-speech translation assessment. It implements an `ASRGenerator` class that wraps CTC-based ASR models from HuggingFace and fairseq, utilizing Torchaudio's CTC decoder for multi-language audio transcription. The toolkit provides a complete pipeline that loads ASR models, transcribes audio, and computes BLEU scores against references using sacrebleu. Developers can use this for evaluating speech translation quality, with examples showing integration with Speechmatrix and Hokkien translation projects. The implementation supports command-line usage with configurable language, audio directory, and reference parameters.

*This is a condensed version that preserves essential implementation details and context.*

# ASR-BLEU Evaluation Toolkit

A toolkit for evaluating speech-to-speech translation systems using ASR models and BLEU scoring.

## Implementation Details

- `ASRGenerator` wraps CTC-based ASR models from HuggingFace and fairseq
- Uses Torchaudio CTC decoder for audio transcription
- Supports multiple languages as defined in `asr_model_cfgs.json`
- Pipeline: loads ASR model → transcribes audio → computes BLEU against references using sacrebleu

## Usage

```bash
python compute_asr_bleu.py --lang <LANG> \
--audio_dirpath <PATH_TO_AUDIO_DIR> \
--reference_path <PATH_TO_REFERENCES_FILE> \
--reference_format txt
```

## Integration Examples

- Used with Speechmatrix project: https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_matrix
- Used with Hokkien speech-to-speech translation: https://github.com/facebookresearch/fairseq/tree/ust/examples/hokkien

## Dependencies

Refer to `requirements.txt` for necessary dependencies.