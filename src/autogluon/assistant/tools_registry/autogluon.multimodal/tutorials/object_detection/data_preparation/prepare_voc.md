Summary: This tutorial covers downloading and preparing the Pascal VOC dataset (VOC2007/VOC2012) for object detection tasks using AutoGluon's MultiModalPredictor. It demonstrates two download methods: AutoGluon CLI commands with dataset name parameters (`-d voc0712`) and optional output paths (`-o ~/data`), plus bash script alternatives. The tutorial explains the expected dataset structure (VOCdevkit containing Annotations, ImageSets, JPEGImages folders) and VOC's XML-based annotation format. Critically, it warns that while VOC format has limited support, COCO format is strongly recommended for better compatibility with AutoGluon. This helps with tasks involving dataset preparation, understanding object detection data formats, and setting up training pipelines for computer vision models.

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712
```


or extract it under a provided output path:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712 --output_path ~/data
```


or make it shorter:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc -o ~/data
```


or download them separately

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```


## Download with Bash Script

You could either extract it under current directory by running:

```
bash download_voc0712.sh
```


or extract it under a provided output path:

```
bash download_voc0712.sh ~/data
```


The command line output will show the progress bar:

```
extract data in current directory
Downloading VOC2007 trainval ...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  438M  100  438M    0     0  92.3M      0  0:00:04  0:00:04 --:--:-- 95.5M
Downloading VOC2007 test data ...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  430M  100  430M    0     0  96.5M      0  0:00:04  0:00:04 --:--:-- 99.1M
Downloading VOC2012 trainval ...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
 73 1907M   73 1401M    0     0   108M      0  0:00:17  0:00:12  0:00:05  118M

```


And after it finished, VOC datasets are extracted in folder `VOCdevkit`, it contains

```
VOC2007  VOC2012
```


And both of them contains:

```
Annotations  ImageSets  JPEGImages  SegmentationClass  SegmentationObject
```


## The VOC Format
VOC also refers to the specific format (in `.xml` file) the VOC dataset is using.

**In Autogluon MultiModalPredictor, we strongly recommend using COCO as your data format instead.
Check [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for more information
about COCO dataset and how to convert a VOC dataset to COCO.**

However, for fast proof testing we also have limit support for VOC format.
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).

## Citation
```
@Article{Everingham10,
   author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
   title = "The Pascal Visual Object Classes (VOC) Challenge",
   journal = "International Journal of Computer Vision",
   volume = "88",
   year = "2010",
   number = "2",
   month = jun,
   pages = "303--338",
}
```
