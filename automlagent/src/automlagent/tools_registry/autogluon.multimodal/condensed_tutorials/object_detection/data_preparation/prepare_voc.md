# Condensed: AutoMM Detection - Prepare Pascal VOC Dataset

Summary: This tutorial provides implementation guidance for preparing the Pascal VOC dataset for object detection tasks using AutoMM. It covers two main implementation approaches: using Python CLI (cross-platform) and Bash scripts (Unix), with specific commands for downloading VOC2007 and VOC2012 datasets. The tutorial helps with dataset setup tasks, including proper directory structure organization and format conversion. Key features include dataset download automation, proper directory structure setup (~8.4GB), and handling both VOC2007 and VOC2012 versions with 20 object categories. It emphasizes COCO format recommendation over VOC format and includes essential information about dataset splits for training and validation.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Pascal VOC Dataset Preparation

## Key Requirements
- Disk space: 8.4 GB
- Recommended: SSD for better performance
- Estimated preparation time: ~10 min on AWS EC2 with EBS

## Dataset Download Options

### 1. Using Python CLI (Cross-platform)
```python
# Download and extract both VOC2007 and VOC2012
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712 --output_path ~/data

# Download separately
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```

### 2. Using Bash Script (Unix systems)
```bash
# Extract in current directory
bash download_voc0712.sh

# Extract to specific path
bash download_voc0712.sh ~/data
```

## Dataset Structure
After extraction:
```
VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── SegmentationObject/
└── VOC2012/
    └── [same structure as VOC2007]
```

## Important Notes and Best Practices

1. **Format Recommendation**:
   - COCO format is strongly recommended over VOC format
   - Refer to "Prepare COCO2017 Dataset" and "Convert Data to COCO Format" tutorials

2. **Dataset Composition**:
   - Training: VOC2007 trainval + VOC2012 trainval (16,551 images)
   - Validation: VOC2007 test
   - Classes: 20 categories (same for both versions)

3. **VOC Format Support**:
   - Limited support available for quick testing
   - Required directories: `Annotations`, `ImageSets`, `JPEGImages`

For customization details, refer to the "Customize AutoMM" documentation.