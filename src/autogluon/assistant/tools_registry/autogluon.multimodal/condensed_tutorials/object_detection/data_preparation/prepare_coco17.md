# Condensed: ```

Summary: This tutorial covers downloading and setting up the COCO2017 dataset for object detection tasks using AutoGluon. It demonstrates two download methods: AutoGluon's CLI tool (`prepare_detection_dataset`) with customizable output paths, and bash scripts for automated downloads. The tutorial helps with dataset preparation tasks including understanding COCO dataset structure (annotations, train/val/test splits totaling ~44GB), working with COCO JSON format for object detection, and converting other formats (like VOC) to COCO format. Key features include CLI command shortcuts, path configuration options, and best practices for using COCO format with AutoGluon MultiModalPredictor for object detection workflows.

*This is a condensed version that preserves essential implementation details and context.*

# COCO Dataset Download and Setup

## Download Methods

### Using AutoGluon CLI
Extract to default `coco17` folder:
```bash
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017
```

Extract to custom path:
```bash
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017 --output_path ~/data
# Short form:
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d coco17 -o ~/data
```

### Using Bash Script
Default location:
```bash
bash download_coco17.sh
```

Custom path:
```bash
bash download_coco17.sh ~/data
```

## Dataset Structure
After download completes, the `coco17` folder contains:
```
annotations  test2017  train2017  unlabeled2017  val2017
```

**Dataset sizes:**
- train2017: ~18GB
- val2017: ~778MB  
- test2017: ~6.2GB
- unlabeled2017: ~19GB

## COCO Format

**Best Practice:** AutoGluon MultiModalPredictor strongly recommends using COCO format (`.json`) for object detection tasks.

**Format Conversion Resources:**
- [Convert Data to COCO Format](convert_data_to_coco_format.ipynb)
- [Convert VOC Format to COCO Format](voc_to_coco.ipynb)

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)

## Citation
```bibtex
@misc{https://doi.org/10.48550/arxiv.1405.0312,
  title = {Microsoft COCO: Common Objects in Context},
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  year = {2014}
}
```