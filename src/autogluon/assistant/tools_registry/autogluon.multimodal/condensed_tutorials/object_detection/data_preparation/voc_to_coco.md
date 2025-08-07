# Condensed: ```

Summary: This tutorial demonstrates how to convert Pascal VOC object detection datasets to COCO format using Python code that helps with data format conversion and custom data splitting for object detection tasks. The tutorial covers: 1) converting pre-defined VOC data splits to COCO format using Python scripts, 2) creating custom data splits with specific train/val/test ratios, and 3) generating COCO-format JSON files from VOC annotations. The tutorial helps with dataset preparation for object detection models by providing Python scripts for format conversion and data splitting with customizable train/val/test ratios.

*This is a condensed version that preserves essential implementation details and context.*

# VOC to COCO Format Conversion

## Converting Pre-defined Splits

VOC datasets typically have this structure:
```
Annotations  ImageSets  JPEGImages
```

With split files under `ImageSets/Main/`:
```
train.txt
val.txt
test.txt
```

Convert these splits to COCO format:
```python
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```

This generates COCO format files in the `Annotations` folder:
```
train_cocoformat.json
val_cocoformat.json
test_cocoformat.json
```

## Custom Data Splitting

Create custom splits with specific ratios without requiring pre-existing split files:

```python
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007 --train_ratio 0.6 --val_ratio 0.2
```

This generates:
```
usersplit_train_cocoformat.json
usersplit_val_cocoformat.json
usersplit_test_cocoformat.json
```

## Additional Resources
- See [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) for more examples
- For customization, refer to [Customize AutoMM](../../advanced_topics/customization.ipynb)