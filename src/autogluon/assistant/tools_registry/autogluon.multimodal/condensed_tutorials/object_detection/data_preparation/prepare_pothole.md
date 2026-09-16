# Condensed: ```

Summary: This tutorial covers AutoGluon's CLI-based dataset preparation for object detection tasks, specifically demonstrating how to download and configure the pothole dataset using the `autogluon.multimodal.cli.prepare_detection_dataset` command with various parameters (dataset name, output path). It provides implementation knowledge for working with COCO-formatted detection datasets, including automatic train/validation/test splitting (3:1:1 ratio) and annotation file structure. The tutorial helps with tasks like setting up object detection datasets, converting VOC format to COCO format, and configuring AutoMM for detection projects. Key features include CLI command syntax variations, dataset format specifications, and best practices emphasizing COCO format over VOC for AutoGluon MultiModalPredictor workflows.

*This is a condensed version that preserves essential implementation details and context.*

# Pothole Dataset Preparation

## Download Dataset

Use AutoGluon's CLI tool to download and prepare the pothole dataset:

```bash
# Basic usage
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole

# With custom output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```

**Dataset Format:** COCO format with 3:1:1 train/validation/test split

**Annotation Files:**
```
pothole/Annotations/usersplit_train_cocoformat.json
pothole/Annotations/usersplit_val_cocoformat.json
pothole/Annotations/usersplit_test_cocoformat.json
```

## Important Notes

⚠️ **If downloading from Kaggle:** Original dataset is in VOC format and unsplit. **AutoGluon strongly recommends COCO format.** See:
- [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb)
- [Convert Data to COCO Format](convert_data_to_coco_format.ipynb)

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