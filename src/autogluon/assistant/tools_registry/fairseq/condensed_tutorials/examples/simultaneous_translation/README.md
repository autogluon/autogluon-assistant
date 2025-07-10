# Condensed: Simultaneous Translation

Summary: This tutorial covers Fairseq's implementations of simultaneous translation models, providing code and configurations for three specific approaches: English-to-Japanese text-to-text wait-k model, English-to-German text-to-text with monotonic multihead attention, and English-to-German speech-to-text simultaneous translation. The documentation helps implement real-time translation systems with different latency-quality tradeoffs. It offers implementation details for specialized attention mechanisms (wait-k, monotonic multihead) and modality-specific considerations for both text-to-text and speech-to-text simultaneous translation tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Simultaneous Translation in Fairseq

Fairseq provides several implementations of simultaneous translation models:

## Available Examples
- **English-to-Japanese text-to-text wait-k model**: [Documentation](docs/enja-waitk.md)
- **English-to-German text-to-text monotonic multihead attention model**: [Documentation](docs/ende-mma.md)
- **English-to-German speech-to-text simultaneous translation model**: [Documentation](../speech_to_text/docs/simulst_mustc_example.md)

*Note: These examples demonstrate different approaches to simultaneous translation with their specific implementations and configurations.*