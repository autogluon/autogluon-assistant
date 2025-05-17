# Condensed: Continuous Training with AutoMM

Summary: This tutorial demonstrates continuous training techniques using AutoMM's MultiModalPredictor, covering three main implementation patterns: extending training with new data, resuming interrupted training from checkpoints, and transfer learning with pre-trained models. It helps with tasks involving model persistence, training continuation, and transfer learning for text, image, and fusion models. Key features include checkpoint management (model.ckpt vs last.ckpt), hyperparameter configuration for different model types (HuggingFace, TIMM, MMDetection), and data consistency requirements. The tutorial emphasizes best practices for production deployment and warns about potential catastrophic forgetting in transfer learning scenarios.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and practices:

# Continuous Training with AutoMM

## Key Implementation Patterns

### 1. Extending Training Time/Adding Data
```python
# Initial training
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)

# Continue training with new data
predictor_2 = MultiModalPredictor.load(model_path)
predictor_2.fit(train_data_2, time_limit=60)
```

**Important**: 
- New data must match the original problem type and classes
- Model checkpoints are saved as `model.ckpt` under `model_path`

### 2. Resuming Interrupted Training
```python
# Resume from last checkpoint
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

**Note**: Uses `last.ckpt` instead of `model.ckpt`

### 3. Transfer Learning with Pre-trained Models

1. Dump existing model:
```python
predictor.dump_model(save_path=dump_model_path)
```

2. Use as foundation for new task:
```python
# For HuggingFace text models
hyperparameters={
    "model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"
}

predictor_new = MultiModalPredictor(label="new_label", path=new_model_path)
predictor_new.fit(
    new_data, 
    hyperparameters=hyperparameters,
    time_limit=30
)
```

**Supported Model Types**:
- HuggingFace text models: `model.hf_text.checkpoint_name`
- TIMM image models: `model.timm_image.checkpoint_name`
- MMDetection models: `model.mmdet_image.checkpoint_name`
- Fusion models combining the above

## Best Practices
1. Set adequate `time_limit` for production (recommended: 1+ hour or `None`)
2. Ensure data consistency when continuing training
3. Consider catastrophic forgetting when transfer learning
4. Verify model checkpoints exist before resuming training

## Warning
Transfer learning may face catastrophic forgetting issues - consider this when applying to new tasks.