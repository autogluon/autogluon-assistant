# Condensed: AutoMM Detection - Finetune on COCO Format Dataset with Customized Settings

Summary: This tutorial demonstrates implementing object detection using AutoGluon's MultiModalPredictor with COCO-format datasets. It covers essential techniques for model initialization, training configuration, and evaluation using YOLOX architectures. Key implementations include two-stage learning rate setup, batch size optimization, validation scheduling, and early stopping mechanisms. The tutorial helps with tasks like configuring object detection models, fine-tuning hyperparameters, and visualizing predictions. Notable features include preset configurations (medium/high/best quality), GPU memory management, CUDA/PyTorch compatibility setup, and performance optimization techniques through learning rate adjustments and validation strategies.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Finetune on COCO Format Dataset

## Key Setup Requirements

```bash
# Critical installations
pip install autogluon.multimodal
pip install -U pip setuptools wheel
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"
```

⚠️ **Important Notes**: 
- Use CUDA 12.4 with PyTorch 2.5
- Install build dependencies before MMCV
- Restart kernel after installation

## Implementation Steps

1. **Initialize Predictor**
```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "yolox_s",  # Using YOLOX-small model
        "env.num_gpus": 1,
    },
    problem_type="object_detection",
    sample_data_path=train_path,
)
```

2. **Model Training**
```python
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optimization.learning_rate": 1e-4,  # Head layers get 100x this value
        "env.per_gpu_batch_size": 16,
        "optimization.max_epochs": 30,
        "optimization.val_check_interval": 1.0,
        "optimization.check_val_every_n_epoch": 3,
        "optimization.patience": 3,
    },
)
```

3. **Quick Implementation Using Presets**
```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",  # Options: medium_quality, high_quality, best_quality
)
predictor.fit(train_path, tuning_data=val_path)
```

4. **Evaluation and Prediction**
```python
# Evaluate model
predictor.evaluate(test_path)

# Make predictions
pred = predictor.predict(test_path)

# Visualize results
visualize_detection(
    pred=pred[12:13],
    detection_classes=predictor.classes,
    conf_threshold=0.25,
    visualization_result_dir="./"
)
```

## Critical Configurations

- **Learning Rate**: Uses two-stage learning rate (head layers get 100x base rate)
- **Batch Size**: Default 16, adjust based on GPU memory
- **Validation**: Checks every 3 epochs
- **Early Stopping**: After 3 consecutive non-improving validations
- **Confidence Threshold**: 0.25 for visualization filtering

## Best Practices

1. Use presets (`medium_quality`, `high_quality`, `best_quality`) for quick implementation
2. Adjust batch size based on available GPU memory
3. Consider larger models for better performance (at cost of speed)
4. Use two-stage learning rate for faster convergence, especially on small datasets
5. Ensure proper CUDA/PyTorch compatibility for MMCV installation