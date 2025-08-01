Summary: This tutorial explains the COCO dataset format for object detection, detailing the required directory structure and JSON schema with specific fields for images, categories, and annotations (including bounding boxes). It covers how to convert VOC format datasets to COCO using AutoGluon's CLI tool, with both custom and predefined dataset splits. The tutorial also mentions conversion options for other formats using tools like FiftyOne. This knowledge is essential for preparing object detection datasets, handling annotation formats, and implementing data conversion pipelines for training and evaluation of object detection models.

```
<dataset_dir>/
    images/
        <imagename0>.<ext>
        <imagename1>.<ext>
        <imagename2>.<ext>
        ...
    annotations/
        train_labels.json
        val_labels.json
        test_labels.json
        ...
```


### 2. Formatting ``*_labels.json``
Below are the key names and value definitions inside ``*_labels.json``:

```javascript
{
    "info": info,
    "licenses": [license], 
    "images": [image],  // list of all images in the dataset
    "annotations": [annotation],  // list of all annotations in the dataset
    "categories": [category]  // list of all categories
}

where:

info = {
    "year": int, 
    "version": str, 
    "description": str, 
    "contributor": str, 
    "url": str, 
    "date_created": datetime,
}

license = {
    "id": int, 
    "name": str, 
    "url": str,
}

image = {
    "id": int, 
    "width": int, 
    "height": int, 
    "file_name": str, 
    "license": int,  // the id of the license
    "date_captured": datetime,
}

category = {
    "id": int, 
    "name": str, 
    "supercategory": str,
}

annotation = {
    "id": int, 
    "image_id": int,  // the id of the image that the annotation belongs to
    "category_id": int,  // the id of the category that the annotation belongs to
    "segmentation": RLE or [polygon], 
    "area": float, 
    "bbox": [x,y,width,height], 
    "iscrowd": int,  // 0 or 1,
}
```


For the sole purpose of running AutoMM, the fields ``"info"`` and ``"licenses"`` are optional. 
``"images"``, ``"categories"``, and ``"annotations"`` are required for training and evaluation, while for prediction only the ``"images"`` field is required.

```json
{
    "info": {...},
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", 
            "id": 1, 
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ...
    ],
    "categories": [
        {"supercategory": "person", "id": 1, "name": "person"},
        {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
        ...
    ],
        
    "images": [
        {
            "license": 4, 
            "file_name": "<imagename0>.<ext>", 
            "height": 427, 
            "width": 640, 
            "date_captured": null, 
            "id": 397133
        },
        ...
    ],
    "annotations": [
        
        ...
    ]
}
```


The following is an example of one sample annotated with COCO format

## Converting VOC format to COCO format
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is a collection of datasets for object detection. 
And VOC format refers to the specific format (in `.xml` file) the Pascal VOC dataset is using.

We have a tutorial guiding you convert your VOC format dataset, i.e. either Pascal VOC Dataset or other datasets in VOC format, to COCO format: [AutoMM Detection - Convert VOC Format Dataset to COCO Format](voc_to_coco.ipynb)

In short, assuming your VOC dataset has the following structure

```
<path_to_VOCdevkit>/
    VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
        labels.txt
    VOC2012/
        Annotations/
        ImageSets/
        JPEGImages/
        labels.txt
    ...
```


### Run the following command:

```python
# If you'd like to customize train/val/test ratio. Note test_ratio = 1 - train_ratio - val_ratio.
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>  
# If you'd like to use the dataset provided train/val/test splits:
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
```


For more details, please see the tutorial: [AutoMM Detection - Convert VOC Format Dataset to COCO Format](voc_to_coco.ipynb).

## Converting other formats to COCO format
We have demonstrated the COCO format and feel free to write your own code to convert your data into the COCO format.
As long as your data conforms to COCO format, it'll work perfectly with the AutoMM pipelines.
In addition, there are a number of 3rd party tools to convert data into COCO format. 
For example, [FiftyOne](https://github.com/voxel51/fiftyone) provides functionalities to convert other formats such as CVAT, YOLO, 
and KITTI etc. into COCO format.
