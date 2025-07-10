Summary: This tutorial covers implementation of speech evaluation metrics for generative spoken language models. It details ASR metrics using word error rate from transcribed speech, ABX metrics for evaluating phonetic category discrimination in quantized representations, and scoring-based metrics (sWUGGY and sBLIMP) from the ZeroSpeech challenge for assessing linguistic properties of speech representations. The tutorial helps with implementing speech quality evaluation, phonetic discrimination testing, and linguistic property assessment of speech models, providing specific techniques for comprehensive speech synthesis evaluation following established benchmarks.

# GSLM Metrics

## ASR Metrics
The suite of metrics here uses an ASR model to transcribe the synthesized speech into text, and then uses text-based metrics. We also use word error rate from ASR transcription itself as one of the metrics. [More details](asr_metrics)

## ABX Metrics
We use [ABX](https://www.semanticscholar.org/paper/ABX-Discriminability-Measures-and-Applications-Schatz/13d3537228f728c1063cc83743cb118bba3367a0) to evaluate how well-separated phonetic categories are with quantized representations. [More details](abx_metrics)

## sWUGGY and sBLIMP
We refer to [ZeroSpeech challenge](https://www.zerospeech.com/2021/track_s.html#scoring-based-metrics) for details on the sWUGGY and sBLIMP metrics.
