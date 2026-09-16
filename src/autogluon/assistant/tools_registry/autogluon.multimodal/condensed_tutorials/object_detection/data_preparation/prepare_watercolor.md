# Condensed: ```

Summary: This tutorial covers downloading and preparing the Watercolor object detection dataset for AutoGluon MultiModalPredictor. It demonstrates two download methods: using AutoGluon's CLI tool (`prepare_detection_dataset`) with dataset name and optional output path parameters, or bash scripts. The tutorial explains the VOC format dataset structure (Annotations, ImageSets, JPEGImages folders) but critically warns that COCO format is strongly recommended over VOC for production use, as VOC has limited support for testing only. Key implementation knowledge includes CLI command syntax, path configuration, and understanding dataset format requirements for AutoGluon's object detection workflows, with guidance on format conversion resources.

*This is a condensed version that preserves essential implementation details and context.*

# Watercolor Dataset Setup

## Download Methods

### Using AutoGluon CLI
```bash
# Extract to current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor

# Extract to custom path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d watercolor -o ~/data
```

### Using Bash Script
```bash
# Extract to current directory
bash download_watercolor.sh

# Extract to custom path
bash download_watercolor.sh ~/data
```

## Dataset Structure

After extraction, the `watercolor` folder contains:
```
Annotations  ImageSets  JPEGImages
```

## Dataset Format

**⚠️ Important:** Watercolor uses VOC format, but **AutoGluon MultiModalPredictor strongly recommends using COCO format instead**. 

- VOC format has limited support for fast testing only
- For VOC datasets, input the root path containing: `Annotations`, `ImageSets`, `JPEGImages`
- See [Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for conversion guidance

## Additional Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)

## Citation
```bibtex
@inproceedings{inoue_2018_cvpr,
    author = {Inoue, Naoto and Furuta, Ryosuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
    title = {Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation},
    booktitle = {CVPR},
    year = {2018}
}
```