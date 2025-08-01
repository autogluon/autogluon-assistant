Summary: "Summary: "Summarize_: "

Summary: "Summarized: "

Summary: "

Summarize: "

Summary: This tutorial provides a tutorial on how to use the AutoGluaMM.multimodal.cli.

Summary: This tutorial demonstrates how to prepare detection datasets for object detection datasets for AutoGludem.

Summary: "

Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "

Summary: This tutorial demonstrates how to prepare object detection datasets using AutoGluon's MultiModal CLI. It covers downloading and preparing the pothole dataset in COCO format, which is the recommended format for AutoGluon's MultiModalPredictor. The tutorial explains command-line options for dataset preparation, including specifying output paths, and notes how the data is automatically split into train/validation/test sets. It warns that the original Kaggle dataset is in VOC format and provides links to additional resources for converting data to COCO format and customizing AutoMM. This knowledge helps with preparing datasets for object detection tasks in AutoGluon.

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
