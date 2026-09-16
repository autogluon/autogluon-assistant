# Condensed: ```

Summary: This tutorial covers converting Pascal VOC format object detection datasets to COCO format using AutoGluon's CLI tool. It demonstrates two key implementations: (1) converting pre-existing VOC dataset splits (train/val/test.txt) to COCO JSON format using the `voc2coco` command with `--root_dir` parameter, and (2) creating custom dataset splits with specified ratios using `--train_ratio` and `--val_ratio` parameters. The tutorial helps with dataset preprocessing tasks for computer vision projects, specifically format conversion between VOC (XML annotations) and COCO (JSON) formats. Key functionality includes automated batch conversion that outputs standardized `*_cocoformat.json` files in the Annotations directory, enabling compatibility with COCO-based training pipelines.

*This is a condensed version that preserves essential implementation details and context.*

# VOC to COCO Format Conversion

## Standard VOC Dataset Structure
```
Annotations  ImageSets  JPEGImages
```

Pre-defined splits in `ImageSets/Main/`:
```
train.txt
val.txt
test.txt
```

## Convert Pre-defined Splits

Convert existing VOC splits to COCO format:

```bash
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```

**Output:** Generates COCO format files in `Annotations/`:
```
train_cocoformat.json
val_cocoformat.json
test_cocoformat.json
```

## Custom Split Ratios

Create custom train/val/test splits (e.g., 0.6/0.2/0.2) without pre-existing split files:

```bash
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007 --train_ratio 0.6 --val_ratio 0.2
```

**Output:** Generates custom splits in `Annotations/`:
```
usersplit_train_cocoformat.json
usersplit_val_cocoformat.json
usersplit_test_cocoformat.json
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)