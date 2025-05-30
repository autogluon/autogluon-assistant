Summary: This tutorial demonstrates how to convert object detection datasets from VOC to COCO format using AutoMM's conversion tool (autogluon.multimodal.cli.voc2coco). It covers two main implementation approaches: converting existing dataset splits and creating custom split ratios. The tutorial explains the required directory structure, command-line usage patterns, and output file formats. Key functionalities include handling predefined VOC splits, creating custom train/val/test ratios, and generating COCO-formatted JSON annotations. This knowledge is particularly useful for data preprocessing tasks in object detection projects using AutoMM, where COCO format is the recommended standard.

# AutoMM Detection - Convert VOC Format Dataset to COCO Format

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/voc_to_coco.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/voc_to_coco.ipynb)



[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is a collection of datasets for object detection. 
And VOC format refers to the specific format (in `.xml` file) the Pascal VOC dataset is using.

In this tutorial, we will convert VOC2007 dataset from VOC format to COCO format. See [AutoMM Detection - Prepare Pascal VOC Dataset](prepare_voc.ipynb) for how to download it.
We will use our tool `voc2coco`. This python script is in our code: 
[voc2coco.py](https://raw.githubusercontent.com/autogluon/autogluon/master/multimodal/src/autogluon/multimodal/cli/voc2coco.py),
and you can also run it as a cli: `python3 -m autogluon.multimodal.cli.voc2coco`.

**Note: In Autogluon MultiModalPredictor, we strongly recommend using COCO as your data format.** However, for fast proof testing we also have limit support for VOC format.

## Convert Existing Splits

Under VOC format root path, we have the following folders:

```
Annotations  ImageSets  JPEGImages
```


And normally there are some pre-defined split files under `ImageSets/Main/`:

```
train.txt
val.txt
test.txt
...
```


We can convert those splits into COCO format by simply running given the root directory, e.g. `./VOCdevkit/VOC2007`:

```
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```


The command line output will show the progress:

```
Start converting !
 17%|█████████████████▍                                                                                  | 841/4952 [00:00<00:00, 15571.88it/s
```


Now those splits are converted to COCO format in `Annotations` folder under the root directory:

```
train_cocoformat.json
val_cocoformat.json
test_cocoformat.json
...
```


## Convert Existing Splits

Instead of using predefined splits, you can also split the data with the train/validation/test ratio you want.
Note that this does not require any pre-existing split files. To split train/validation/test by 0.6/0.2/0.2, run:

```
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007 --train_ratio 0.6 --val_ratio 0.2
```


The command line output will show the progress:

```
Start converting !
 17%|█████████████████▍                                                                                  | 841/4952 [00:00<00:00, 15571.88it/s
```


And this will generate user splited COCO format in `Annotations` folder under the root directory:

```
usersplit_train_cocoformat.json
usersplit_val_cocoformat.json
usersplit_test_cocoformat.json
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).
