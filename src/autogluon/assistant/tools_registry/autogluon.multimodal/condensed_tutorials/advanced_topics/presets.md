# Condensed: AutoMM Presets

Summary: This tutorial demonstrates how to use AutoMM's preset configurations (medium_quality, high_quality, best_quality) for automated machine learning tasks. It provides implementation details for configuring MultiModalPredictor with different performance-speed tradeoffs, including HPO variants for hyperparameter optimization. The tutorial covers essential code patterns for model setup, training, and evaluation, with specific focus on preset selection, resource management, and tunable parameters (model backbone, batch size, learning rate, max epoch, optimizer type). It's particularly useful for tasks requiring automated model configuration and performance optimization, helping developers choose appropriate presets based on their computational resources and performance requirements.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Presets Tutorial

## Key Concepts
AutoMM provides three preset configurations to simplify hyperparameter setup:
- `medium_quality`: Fast training/inference, smaller models
- `high_quality`: Balanced performance/speed
- `best_quality`: Optimal performance, higher computational requirements
- HPO variants: Add `_hpo` suffix (e.g., `medium_quality_hpo`) for hyperparameter optimization

## Implementation

### Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
import warnings
warnings.filterwarnings('ignore')
```

### Using Presets

1. Medium Quality (Fast)
```python
predictor = MultiModalPredictor(
    label='label', 
    eval_metric='acc', 
    presets="medium_quality"
)
predictor.fit(
    train_data=train_data,
    time_limit=20  # seconds
)
```

2. High Quality (Balanced)
```python
predictor = MultiModalPredictor(
    label='label', 
    eval_metric='acc', 
    presets="high_quality"
)
predictor.fit(
    train_data=train_data,
    time_limit=20
)
```

3. Best Quality (Performance)
```python
predictor = MultiModalPredictor(
    label='label', 
    eval_metric='acc', 
    presets="best_quality"
)
predictor.fit(
    train_data=train_data,
    time_limit=180
)
```

### Viewing Preset Configurations
```python
from autogluon.multimodal.presets import get_automm_presets

# Get preset details
hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(
    problem_type="default", 
    presets="high_quality"
)
```

## Important Notes

1. **Resource Requirements**:
   - Best quality preset requires high-end GPUs with large memory
   - Adjust time_limit based on dataset size and preset complexity

2. **HPO Tunable Parameters**:
   - Model backbone
   - Batch size
   - Learning rate
   - Max epoch
   - Optimizer type

3. **Performance vs Speed**:
   - medium_quality → fastest, lowest performance
   - high_quality → balanced option
   - best_quality → slowest, highest performance

4. **Evaluation**:
```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

For customization options, refer to the Customize AutoMM documentation.