# Condensed: ```

Summary: This tutorial demonstrates how to prepare the Watercolor object detection dataset using AutoGluon. It covers two implementation methods: using AutoGluon CLI commands or a bash script, with options for specifying output paths. The tutorial explains that Watercolor uses VOC format but recommends converting to COCO format for use with AutoGluon MultiModalPredictor. It references additional resources for COCO format conversion and customization. This knowledge helps with dataset preparation tasks for object detection models, specifically working with the Watercolor dataset in AutoGluon's ecosystem.

*This is a condensed version that preserves essential implementation details and context.*

# Preparing Detection Dataset - Watercolor

## Download Methods

### Using AutoGluon CLI

```python
# Basic usage - extracts to current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor

# Specify output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d watercolor -o ~/data
```

### Using Bash Script

```bash
# Extract to current directory
bash download_watercolor.sh

# Extract to specific path
bash download_watercolor.sh ~/data
```

## Dataset Format

Watercolor uses VOC format with the following structure:
```
Annotations  ImageSets  JPEGImages
```

**Important:** AutoGluon MultiModalPredictor strongly recommends using COCO format instead of VOC. See [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for conversion instructions.

## Additional Resources
- For more examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization: [Customize AutoMM](../../advanced_topics/customization.ipynb)

## Citation
```
@inproceedings{inoue_2018_cvpr,
    author = {Inoue, Naoto and Furuta, Ryosuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
    title = {Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```