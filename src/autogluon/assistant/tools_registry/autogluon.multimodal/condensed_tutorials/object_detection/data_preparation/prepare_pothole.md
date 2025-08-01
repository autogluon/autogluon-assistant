# Condensed: ```

Summary: "Summary: "Summarize_: "

Summary: "Summarized: "

Summary: "

Summarize: "

Summary: This tutorial provides a tutorial on how to use the AutoGluaMM.multimodal.cli.

Summary: This tutorial demonstrates how to prepare detection datasets for object detection datasets for AutoGludem.

Summary: "

Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "Summary: "

Summary: This tutorial demonstrates how to prepare object detection datasets using AutoGluon's MultiModal CLI. It covers downloading and preparing the pothole dataset in COCO format, which is the recommended format for AutoGluon's MultiModalPredictor. The tutorial explains command-line options for dataset preparation, including specifying output paths, and notes how the data is automatically split into train/validation/test sets. It warns that the original Kaggle dataset is in VOC format and provides links to additional resources for converting data to COCO format and customizing AutoMM. This knowledge helps with preparing datasets for object detection tasks in AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

# Preparing Detection Dataset

## Download and Prepare Dataset

Download the pothole dataset in COCO format:

```bash
# Basic usage
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole

# Specify output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```

The dataset is automatically split into train/validation/test sets (3:1:1 ratio) with annotation files at:
```
pothole/Annotations/usersplit_train_cocoformat.json
pothole/Annotations/usersplit_val_cocoformat.json
pothole/Annotations/usersplit_test_cocoformat.json
```

## Important Note for Kaggle Downloads

**Warning:** The original Pothole dataset from Kaggle is in VOC format and not pre-split. It's strongly recommended to use COCO format with AutoGluon MultiModalPredictor.

For more information:
- See [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb)
- See [Convert Data to COCO Format](convert_data_to_coco_format.ipynb)

## Additional Resources
- For more examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization: [Customize AutoMM](../../advanced_topics/customization.ipynb)