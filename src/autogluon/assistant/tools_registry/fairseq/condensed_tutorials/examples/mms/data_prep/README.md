# Condensed: Data Preparation

Summary: This tutorial provides implementation steps for audio alignment and segmentation using torchaudio and uroman. It covers installing dependencies, creating transcript files, and running forced alignment to generate segmented audio files with corresponding metadata. The tutorial helps with tasks like splitting audio based on text segments, generating timestamps, and processing non-English audio through universal romanization. Key features include cross-language support, output of normalized text and uroman tokens, and generation of a detailed manifest.json file containing segment information that can be visualized using NeMo's Speech Data Explorer.

*This is a condensed version that preserves essential implementation details and context.*

# Data Preparation

## Implementation Steps for Audio Alignment and Segmentation

### Step 1: Install torchaudio (nightly version)
```bash
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

### Step 2: Download uroman (universal romanizer)
```bash
git clone git@github.com:isi-nlp/uroman.git
```

### Step 3: Install dependencies
```bash
apt install sox 
pip install sox dataclasses
```

### Step 4: Create transcript file
Create a text file with each line representing a desired audio segment:
```
Text of the desired first segment
Text of the desired second segment
Text of the desired third segment
```

### Step 5: Run forced alignment
```bash
python align_and_segment.py --audio /path/to/audio.wav --text_filepath /path/to/textfile --lang <iso> --outdir /path/to/output --uroman /path/to/uroman/bin
```

## Output
The script generates:
- Segmented audio files in the output directory
- A `manifest.json` file containing:
  - Audio segment paths
  - Timestamps
  - Transcripts
  - Normalized text
  - Uroman tokens

Example manifest.json:
```json
{"audio_start_sec": 0.0, "audio_filepath": "/path/to/output/segment1.flac", "duration": 6.8, "text": "she wondered afterwards how she could have spoken with that hard serenity how she could have", "normalized_text": "she wondered afterwards how she could have spoken with that hard serenity how she could have", "uroman_tokens": "s h e w o n d e r e d a f t e r w a r d s h o w s h e c o u l d h a v e s p o k e n w i t h t h a t h a r d s e r e n i t y h o w s h e c o u l d h a v e"}
```

**Key Feature**: The alignment model works with non-English audio as it outputs uroman tokens for input audio in any language.

**Visualization**: Use [Speech Data Explorer](https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer) from NeMo toolkit to visualize the segmented audio files.