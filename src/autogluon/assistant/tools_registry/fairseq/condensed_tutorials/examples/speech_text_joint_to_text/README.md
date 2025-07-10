# Condensed: Joint Speech Text training in Fairseq

Summary: This tutorial presents a Fairseq s2t extension for joint speech-text training, enabling developers to implement unified speech-text models. It covers techniques for multi-task learning that leverages text data to enhance speech-to-text tasks like translation and recognition. Key functionalities include speech-text pre-training, auxiliary text translation integration, and a multi-task learning framework. The implementation provides concrete examples for English-German translation using MuST-C, multilingual speech translation for IWSLT 2021, and speech-text joint pre-training, all built upon Fairseq's sequence modeling toolkit.

*This is a condensed version that preserves essential implementation details and context.*

# Joint Speech Text Training in Fairseq

An extension of Fairseq s2t that enhances speech-to-text tasks with co-trained text-to-text mapping tasks.

## Implementation Examples
- [English-to-German MuST-C model](docs/ende-mustc.md)
- [IWSLT 2021 Multilingual Speech Translation](docs/iwslt2021.md)
- [Speech Text Joint Pre-training](docs/pre-training.md)

## Key Features
- Unified speech-text pre-training for translation and recognition
- Multi-task learning framework leveraging text data for speech-to-text tasks
- Integration with fairseq's extensible toolkit for sequence modeling

## Technical References
The implementation builds upon research from multiple papers focusing on:
- Unified speech-text pre-training
- Auxiliary text translation for improving speech translation
- Multi-task learning frameworks for speech-to-text tasks

For detailed implementation specifics, refer to the example documentation links provided above.