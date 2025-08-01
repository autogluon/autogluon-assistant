# Condensed: ```

Summary: This tutorial demonstrates how to download and use the VOC dataset for object detection tasks with AutoGluon. It covers multiple download methods: using AutoGluon CLI commands to get the full VOC0712 dataset or individual VOC2007/2012 datasets, and using bash scripts. The tutorial explains the resulting dataset structure with folders like Annotations, ImageSets, and JPEGImages. It emphasizes that AutoGluon MultiModalPredictor recommends using COCO format instead of VOC format for better compatibility, though limited VOC format support exists for testing. The guide includes the necessary folder structure requirements when using VOC format and points to additional resources for examples and customization.

*This is a condensed version that preserves essential implementation details and context.*

# Downloading and Using VOC Dataset for Object Detection

## Download Using AutoGluon CLI

```python
# Download full VOC0712 dataset to current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712

# Download to specific path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712 --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc -o ~/data

# Download VOC2007 and VOC2012 separately
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```

## Download Using Bash Script

```bash
# Extract to current directory
bash download_voc0712.sh

# Extract to specific path
bash download_voc0712.sh ~/data
```

## Dataset Structure

After downloading, the VOC dataset will be extracted to a `VOCdevkit` folder containing:
```
VOC2007/  VOC2012/
```

Each containing:
```
Annotations/  ImageSets/  JPEGImages/  SegmentationClass/  SegmentationObject/
```

## Important Note on Format

**Warning: AutoGluon MultiModalPredictor strongly recommends using COCO format instead of VOC format.** 
See "AutoMM Detection - Prepare COCO2017 Dataset" and "Convert Data to COCO Format" tutorials for more information.

For testing purposes, limited VOC format support is available. When using VOC format, the input path must contain at least:
```
Annotations/  ImageSets/  JPEGImages/
```

## Additional Resources
- For more examples, see [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization, refer to [Customize AutoMM](../../advanced_topics/customization.ipynb)