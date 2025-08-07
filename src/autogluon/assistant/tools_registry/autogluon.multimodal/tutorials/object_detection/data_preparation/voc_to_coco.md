Summary: This tutorial demonstrates how to convert Pascal VOC object detection datasets to COCO format using Python code that helps with data format conversion and custom data splitting for object detection tasks. The tutorial covers: 1) converting pre-defined VOC data splits to COCO format using Python scripts, 2) creating custom data splits with specific train/val/test ratios, and 3) generating COCO-format JSON files from VOC annotations. The tutorial helps with dataset preparation for object detection models by providing Python scripts for format conversion and data splitting with customizable train/val/test ratios.

```
Annotations  ImageSets  JPEGImages
```


And normally there are some pre-defined split files under `ImageSets/Main/`:

```
train.txt
val.txt
test.txt
...
```


We can convert those splits into COCO format by simply running given the root directory, e.g. `./VOCdevkit/VOC2007`:

```
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```


The command line output will show the progress:

```
Start converting !
 17%|█████████████████▍                                                                                  | 841/4952 [00:00<00:00, 15571.88it/s
```


Now those splits are converted to COCO format in `Annotations` folder under the root directory:

```
train_cocoformat.json
val_cocoformat.json
test_cocoformat.json
...
```


## Convert Existing Splits

Instead of using predefined splits, you can also split the data with the train/validation/test ratio you want.
Note that this does not require any pre-existing split files. To split train/validation/test by 0.6/0.2/0.2, run:

```
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007 --train_ratio 0.6 --val_ratio 0.2
```


The command line output will show the progress:

```
Start converting !
 17%|█████████████████▍                                                                                  | 841/4952 [00:00<00:00, 15571.88it/s
```


And this will generate user splited COCO format in `Annotations` folder under the root directory:

```
usersplit_train_cocoformat.json
usersplit_val_cocoformat.json
usersplit_test_cocoformat.json
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).
