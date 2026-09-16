# Condensed: ```

Summary: This tutorial covers downloading and preparing the Pascal VOC dataset (VOC2007/VOC2012) for object detection tasks using AutoGluon's MultiModalPredictor. It demonstrates two download methods: AutoGluon CLI commands with dataset name parameters (`-d voc0712`) and optional output paths (`-o ~/data`), plus bash script alternatives. The tutorial explains the expected dataset structure (VOCdevkit containing Annotations, ImageSets, JPEGImages folders) and VOC's XML-based annotation format. Critically, it warns that while VOC format has limited support, COCO format is strongly recommended for better compatibility with AutoGluon. This helps with tasks involving dataset preparation, understanding object detection data formats, and setting up training pipelines for computer vision models.

*This is a condensed version that preserves essential implementation details and context.*

# VOC Dataset Preparation - Condensed

## Download Methods

### Using AutoGluon CLI

**Basic command:**
```bash
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712
```

**With custom output path:**
```bash
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712 --output_path ~/data
# Short form:
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc -o ~/data
```

**Separate downloads:**
```bash
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```

### Using Bash Script

```bash
bash download_voc0712.sh          # Current directory
bash download_voc0712.sh ~/data   # Custom path
```

## Dataset Structure

After extraction, `VOCdevkit` folder contains:
```
VOC2007  VOC2012
```

Each contains:
```
Annotations  ImageSets  JPEGImages  SegmentationClass  SegmentationObject
```

## VOC Format

⚠️ **Important:** VOC format uses `.xml` files. **AutoGluon strongly recommends using COCO format instead** for better compatibility.

For VOC format usage, the input path must contain minimum:
```
Annotations  ImageSets  JPEGImages
```

**See:** [Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb)

## References

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)