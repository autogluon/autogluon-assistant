# Condensed: ```

Summary: This tutorial covers COCO format implementation for object detection datasets, including complete JSON schema specifications with required fields (images, annotations, categories) and optional fields (info, licenses). It provides detailed object definitions for image metadata, category hierarchies, and annotation structures with bounding boxes, segmentation masks, and relational IDs. The tutorial helps with: (1) structuring object detection datasets in COCO format, (2) creating properly formatted JSON annotation files, (3) converting Pascal VOC datasets to COCO using AutoGluon's CLI tool with customizable train/val/test splits, and (4) integrating other formats (CVAT, YOLO, KITTI) via third-party tools like FiftyOne for AutoMM pipeline compatibility.

*This is a condensed version that preserves essential implementation details and context.*

# COCO Format for Object Detection

## Dataset Structure
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
```

## JSON Format Specification

### Core Structure
```javascript
{
    "info": info,                    // Optional for AutoMM
    "licenses": [license],           // Optional for AutoMM
    "images": [image],               // Required for train/eval/predict
    "annotations": [annotation],     // Required for train/eval only
    "categories": [category]         // Required for train/eval only
}
```

### Key Object Definitions

```javascript
image = {
    "id": int, 
    "width": int, 
    "height": int, 
    "file_name": str, 
    "license": int,
    "date_captured": datetime,
}

category = {
    "id": int, 
    "name": str, 
    "supercategory": str,
}

annotation = {
    "id": int, 
    "image_id": int,        // Links to image.id
    "category_id": int,     // Links to category.id
    "segmentation": RLE or [polygon], 
    "area": float, 
    "bbox": [x,y,width,height], 
    "iscrowd": int,        // 0 or 1
}
```

**Important**: For AutoMM, `"info"` and `"licenses"` are optional. Prediction requires only `"images"`, while training/evaluation needs `"images"`, `"categories"`, and `"annotations"`.

## Converting VOC to COCO Format

### VOC Dataset Structure
```
<path_to_VOCdevkit>/
    VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
        labels.txt
```

### Conversion Command
```python
# Custom train/val/test split:
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>

# Use existing dataset splits:
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
```

## Converting Other Formats

Write custom conversion code or use third-party tools like [FiftyOne](https://github.com/voxel51/fiftyone) to convert formats (CVAT, YOLO, KITTI) to COCO. Any data conforming to COCO format works with AutoMM pipelines.