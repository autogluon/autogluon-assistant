Summary: This tutorial provides implementation details for converting object detection datasets to COCO format, essential for training deep learning models. It covers the specific JSON structure requirements for COCO format, including mandatory fields (images, annotations, categories) and their detailed specifications. The tutorial demonstrates how to organize directory structures, handle VOC to COCO conversion using AutoGluon's CLI tools, and mentions alternative conversion options using FiftyOne for CVAT, YOLO, and KITTI formats. Key features include bounding box coordinate formatting, image metadata specifications, and annotation requirements, making it valuable for data preprocessing in object detection tasks.

# Converting Data to COCO Format for Object Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/convert_data_to_coco_format.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/convert_data_to_coco_format.ipynb)

The COCO (Common Objects in Context) dataset format has become the de facto standard for object detection tasks. In AutoGluon's object detection pipeline, we use the COCO format as our standard data format for both training and inference. This tutorial will guide you through preparing your data in COCO format.

## How to Prepare COCO Format Data
### 1. Directory Structure Requirements
Your dataset should be organized following this structure:

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

### 2. JSON Structure Requirements
The annotation files (`*_labels.json`) must contain the following components:

```javascript
{
    "info": info,                    // Dataset metadata (optional)
    "licenses": [license],           // Licensing information (optional)
    "images": [image],              // List of image metadata
    "annotations": [annotation],     // List of object annotations
    "categories": [category]         // List of object categories
}

where:

info = {
    "year": int,                    // Dataset creation year
    "version": str,                 // Version information
    "description": str,             // Dataset description
    "contributor": str,             // Dataset contributor
    "url": str,                     // Related URL
    "date_created": datetime        // Creation date
}

license = {
    "id": int,                      // License identifier
    "name": str,                    // License name
    "url": str                      // License URL
}

image = {
    "id": int,                      // Unique image identifier
    "width": int,                   // Image width in pixels
    "height": int,                  // Image height in pixels
    "file_name": str,               // Image file name
    "license": int,                 // Reference to license id
    "date_captured": datetime       // Image capture date
}

category = {
    "id": int,                      // Unique category identifier
    "name": str,                    // Category name
    "supercategory": str           // Parent category name
}

annotation = {
    "id": int,                      // Unique annotation identifier
    "image_id": int,                // Reference to image
    "category_id": int,             // Reference to category
    "segmentation": RLE or [polygon], // Segmentation data
    "area": float,                  // Object area in pixels
    "bbox": [x,y,width,height],     // Bounding box coordinates
    "iscrowd": int                  // Instance vs group annotation flag (0 or 1)
}
```

**Important Note:** For AutoGluon's object detection tasks, only the "images", "categories", and "annotations" fields are required. The "info" and "licenses" fields are optional for training and evaluation. For prediction, only the "images" field is required.

Here's an example of a COCO format annotation file:

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

## Converting VOC Format to COCO Format

[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is another widely used dataset format for object detection. If your data is in VOC format (using .xml files), you can easily convert it to COCO format using AutoGluon's conversion tools.

For a comprehensive guide on converting VOC format datasets to COCO format, refer to: [AutoMM Detection - Convert VOC Format Dataset to COCO Format](voc_to_coco.ipynb)

Your VOC dataset should have the following structure:

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

### Conversion Commands

```python
# To convert with custom dataset splits (test_ratio = 1 - train_ratio - val_ratio):
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>

# To use predefined dataset splits:
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
```

## Converting Other Formats to COCO

While this tutorial focuses on VOC to COCO conversion, you can convert other formats to COCO format either by:
1. Writing your own conversion scripts following the COCO format specification
2. Using third-party tools like [FiftyOne](https://github.com/voxel51/fiftyone), which provides converters for various formats including:
   - CVAT
   - YOLO
   - KITTI
   - And more

As long as your converted data follows the COCO format specification detailed above, it will be fully compatible with AutoGluon's object detection pipeline.
