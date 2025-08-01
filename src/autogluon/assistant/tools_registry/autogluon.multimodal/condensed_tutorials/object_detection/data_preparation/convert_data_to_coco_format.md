# Condensed: ```

Summary: This tutorial explains the COCO dataset format for object detection, detailing the required directory structure and JSON schema with specific fields for images, categories, and annotations (including bounding boxes). It covers how to convert VOC format datasets to COCO using AutoGluon's CLI tool, with both custom and predefined dataset splits. The tutorial also mentions conversion options for other formats using tools like FiftyOne. This knowledge is essential for preparing object detection datasets, handling annotation formats, and implementing data conversion pipelines for training and evaluation of object detection models.

*This is a condensed version that preserves essential implementation details and context.*

# Dataset Format for Object Detection

## Directory Structure
```
<dataset_dir>/
    images/
        <imagename0>.<ext>
        <imagename1>.<ext>
        ...
    annotations/
        train_labels.json
        val_labels.json
        test_labels.json
        ...
```

## COCO Format JSON Structure
Required JSON structure for `*_labels.json`:

```javascript
{
    "info": info,  // optional
    "licenses": [license],  // optional
    "images": [image],  // required - list of all images
    "annotations": [annotation],  // required for training/evaluation
    "categories": [category]  // required for training/evaluation
}
```

### Key Components

```javascript
image = {
    "id": int, 
    "width": int, 
    "height": int, 
    "file_name": str, 
    "license": int,  // license id
    "date_captured": datetime,
}

category = {
    "id": int, 
    "name": str, 
    "supercategory": str,
}

annotation = {
    "id": int, 
    "image_id": int,  // image id this annotation belongs to
    "category_id": int,  // category id this annotation belongs to
    "segmentation": RLE or [polygon], 
    "area": float, 
    "bbox": [x,y,width,height], 
    "iscrowd": int,  // 0 or 1
}
```

## Converting VOC Format to COCO Format

### VOC Directory Structure
```
<path_to_VOCdevkit>/
    VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
        labels.txt
    VOC2012/
        ...
```

### Conversion Command
```python
# Custom train/val/test split:
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>

# Use dataset's provided splits:
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
```

For more details, see tutorial: [AutoMM Detection - Convert VOC Format Dataset to COCO Format](voc_to_coco.ipynb).

## Converting Other Formats
You can write custom code to convert your data to COCO format or use third-party tools like [FiftyOne](https://github.com/voxel51/fiftyone) which supports converting formats such as CVAT, YOLO, and KITTI to COCO format.