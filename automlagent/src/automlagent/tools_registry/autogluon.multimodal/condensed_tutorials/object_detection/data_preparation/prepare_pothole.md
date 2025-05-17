# Condensed: AutoMM Detection - Prepare Pothole Dataset

Summary: This tutorial demonstrates how to prepare a pothole detection dataset for AutoMM Detection, specifically focusing on implementing dataset preparation using CLI tools and understanding dataset structure. It helps with tasks related to setting up object detection training data, particularly for pothole detection use cases. Key features include CLI-based dataset preparation, COCO format conversion, and dataset splitting (3:1:1 ratio). The tutorial covers essential implementation details like storage requirements, execution time estimates, and best practices for using AutoMM MultiModalPredictor with COCO-formatted data, making it valuable for developers working on computer vision detection tasks.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Pothole Dataset Preparation

## Key Requirements
- Disk space: 1 GB
- Preferred storage: SSD
- Typical preparation time: ~3 min on AWS EC2 with EBS

## Dataset Overview
- 665 images of potholes on roads
- Used for AutoMM Detection fine-tuning demonstrations
- Available formats: COCO (recommended) or VOC

## Implementation Methods

### 1. Using CLI Tool (Recommended)
```bash
# Basic usage - current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole

# With custom output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```

### 2. Dataset Structure
After preparation, COCO format annotations are split (3:1:1 ratio):
```
pothole/Annotations/
├── usersplit_train_cocoformat.json
├── usersplit_val_cocoformat.json
└── usersplit_test_cocoformat.json
```

## Important Notes
- **Strongly recommended**: Use COCO format instead of VOC
- For VOC to COCO conversion, refer to:
  - "AutoMM Detection - Prepare COCO2017 Dataset"
  - "Convert Data to COCO Format"

## Best Practices
1. Always use COCO format for AutoMM MultiModalPredictor
2. Verify dataset splits after preparation
3. Use SSD for better performance

For customization options, refer to the "Customize AutoMM" documentation.