# Condensed: AutoMM Detection - Convert VOC Format Dataset to COCO Format

Summary: This tutorial demonstrates how to convert object detection datasets from VOC to COCO format using AutoMM's conversion tool (autogluon.multimodal.cli.voc2coco). It covers two main implementation approaches: converting existing dataset splits and creating custom split ratios. The tutorial explains the required directory structure, command-line usage patterns, and output file formats. Key functionalities include handling predefined VOC splits, creating custom train/val/test ratios, and generating COCO-formatted JSON annotations. This knowledge is particularly useful for data preprocessing tasks in object detection projects using AutoMM, where COCO format is the recommended standard.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Converting VOC Format to COCO Format for AutoMM Detection

## Key Points
- AutoMM strongly recommends using COCO format over VOC format
- Conversion tool: `autogluon.multimodal.cli.voc2coco`

## Implementation Details

### Directory Structure
```
VOCdevkit/VOC2007/
├── Annotations/
├── ImageSets/Main/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── JPEGImages/
```

### Converting Existing Splits
```bash
# Basic conversion using existing splits
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```

Output files:
- `Annotations/train_cocoformat.json`
- `Annotations/val_cocoformat.json`
- `Annotations/test_cocoformat.json`

### Custom Split Ratios
```bash
# Create custom splits (60/20/20)
python3 -m autogluon.multimodal.cli.voc2coco \
    --root_dir ./VOCdevkit/VOC2007 \
    --train_ratio 0.6 \
    --val_ratio 0.2
```

Output files:
- `Annotations/usersplit_train_cocoformat.json`
- `Annotations/usersplit_val_cocoformat.json`
- `Annotations/usersplit_test_cocoformat.json`

## Best Practices
1. Use COCO format for optimal compatibility with AutoMM
2. VOC format is supported but only for quick testing
3. Choose between existing splits or custom ratios based on your needs

For customization details, refer to the AutoMM customization documentation.