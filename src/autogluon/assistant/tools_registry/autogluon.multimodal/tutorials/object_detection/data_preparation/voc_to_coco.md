Summary: This tutorial covers converting Pascal VOC format object detection datasets to COCO format using AutoGluon's CLI tool. It demonstrates two key implementations: (1) converting pre-existing VOC dataset splits (train/val/test.txt) to COCO JSON format using the `voc2coco` command with `--root_dir` parameter, and (2) creating custom dataset splits with specified ratios using `--train_ratio` and `--val_ratio` parameters. The tutorial helps with dataset preprocessing tasks for computer vision projects, specifically format conversion between VOC (XML annotations) and COCO (JSON) formats. Key functionality includes automated batch conversion that outputs standardized `*_cocoformat.json` files in the Annotations directory, enabling compatibility with COCO-based training pipelines.

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
