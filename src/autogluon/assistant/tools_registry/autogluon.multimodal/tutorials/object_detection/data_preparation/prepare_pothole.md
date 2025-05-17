Summary: This tutorial demonstrates how to prepare a pothole detection dataset for AutoMM Detection, specifically focusing on implementing dataset preparation using CLI tools and understanding dataset structure. It helps with tasks related to setting up object detection training data, particularly for pothole detection use cases. Key features include CLI-based dataset preparation, COCO format conversion, and dataset splitting (3:1:1 ratio). The tutorial covers essential implementation details like storage requirements, execution time estimates, and best practices for using AutoMM MultiModalPredictor with COCO-formatted data, making it valuable for developers working on computer vision detection tasks.

# AutoMM Detection - Prepare Pothole Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/prepare_pothole.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/prepare_pothole.ipynb)



[Pothole](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) is a small object detection dataset with 665 images,
and has a specific domain, i.e. potholes on the road. This dataset will be used to show how to [AutoMM Detection - Fast Finetune on COCO Format Dataset](../finetune/detection_fast_finetune_coco.ipynb) and [AutoMM Detection - High Performance Finetune on COCO Format Dataset](../finetune/detection_high_performance_finetune_coco.ipynb).

You need 1 GB disk space to download and extract this dataset. SSD is preferred over HDD because of its better performance.
The total time to prepare the dataset depends on your Internet speed and disk performance. For example, it often takes 3 min on AWS EC2 with EBS.

You can download the dataset from its [kaggle page](https://www.kaggle.com/datasets/andrewmvd/pothole-detection).
Or you can also use our cli tool `prepare_detection_dataset` that can download all datasets mentioned in our tutorials.
This python script is in our code: 
[prepare_detection_dataset.py](https://raw.githubusercontent.com/autogluon/autogluon/master/multimodal/src/autogluon/multimodal/cli/prepare_detection_dataset.py),
and you can also run it as a cli: `python3 -m autogluon.multimodal.cli.prepare_detection_dataset`.

## Download with Python Script

You could either extract it in pothole folder under current directory by running:

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
