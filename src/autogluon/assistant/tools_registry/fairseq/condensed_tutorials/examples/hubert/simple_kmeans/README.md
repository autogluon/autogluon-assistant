# Condensed: Sharded Feature Extraction and K-means Application

Summary: This tutorial demonstrates how to implement HUBERT label preparation through a sharded processing pipeline. It covers techniques for efficient feature extraction (both MFCC and HUBERT features) from audio files, k-means clustering for creating discrete speech units, and applying trained k-means models to generate labels. Key functionalities include handling large datasets through sharding, memory-efficient processing with chunking options, transformer layer feature extraction, configurable data sampling for k-means training, and merging distributed results. This knowledge is valuable for implementing speech representation learning pipelines, self-supervised audio preprocessing, and discrete token generation from continuous speech features.

*This is a condensed version that preserves essential implementation details and context.*

# Sharded Feature Extraction and K-means Application

## Overview
This tutorial covers preparing HUBERT labels from tsv files through:
1. Feature extraction
2. K-means clustering
3. K-means application

## Data Preparation
TSV files contain audio paths with this format:
```
<root-dir>
<audio-path-1>
<audio-path-2>
...
```

## Feature Extraction

### MFCC Features
Extract 39-D MFCC+delta+ddelta features for initial HUBERT training:
```sh
python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
```
- Shards the TSV file into `${nshard}` parts
- Processes the `${rank}`-th shard (where rank ∈ [0, nshard-1])
- Saves features to `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`

### HUBERT Features
Extract features from a trained HUBERT model:
```sh
python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
```
- Extracts from the specified transformer `${layer}`
- Use `--max_chunk` parameter to decrease chunk size if OOM occurs

## K-means Clustering
Fit a k-means model:
```sh
python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1
```
- Default samples 10% of data
- Use `--percent -1` to use all data
- Model saved to `${km_path}`

## K-means Application
Apply the trained k-means model to get labels:
```sh
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
```
- Processes the `${rank}`-th shard
- Saves labels to `${lab_dir}/${split}_${rank}_${shard}.km`

### Merge Shards
```sh
for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
```

### Create Dictionary
```sh
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt
```