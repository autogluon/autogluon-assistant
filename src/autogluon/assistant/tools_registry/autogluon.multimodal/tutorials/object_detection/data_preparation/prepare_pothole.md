Summary: This tutorial covers AutoGluon's CLI-based dataset preparation for object detection tasks, specifically demonstrating how to download and configure the pothole dataset using the `autogluon.multimodal.cli.prepare_detection_dataset` command with various parameters (dataset name, output path). It provides implementation knowledge for working with COCO-formatted detection datasets, including automatic train/validation/test splitting (3:1:1 ratio) and annotation file structure. The tutorial helps with tasks like setting up object detection datasets, converting VOC format to COCO format, and configuring AutoMM for detection projects. Key features include CLI command syntax variations, dataset format specifications, and best practices emphasizing COCO format over VOC for AutoGluon MultiModalPredictor workflows.

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole
```


or extract it in pothole folder under a provided output path:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole --output_path ~/data
```


or make it shorter:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```


The dataset downloaded with the provided tool is in COCO format and split to train/validation/test set with ratio 3:1:1.
And the annotation files are:

```
pothole/Annotations/usersplit_train_cocoformat.json
pothole/Annotations/usersplit_val_cocoformat.json
pothole/Annotations/usersplit_test_cocoformat.json
```


## If You Download From Kaggle

Original Pothole dataset is in VOC format and is not split. **In Autogluon MultiModalPredictor, we strongly recommend using COCO as your data format instead.
Check [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for more information
about COCO dataset and how to split and convert a VOC dataset to COCO.**


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).

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
