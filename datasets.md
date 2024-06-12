# Downloading and organizing data

## Downloading

### ImageNet V2

```
wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz
```

### ImageNet-R

```
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvzf imagenet-r.tar
```

### ObjectNet

```
wget https://objectnet.dev/downloads/objectnet-1.0.zip
unzip objectnet-1.0.zip
```

### ImageNet Sketch
See the download links [here](https://github.com/HaohanWang/ImageNet-Sketch).

### ImageNet-A

```
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvzf imagenet-a.tar
```

## Organizing


Put all downloaded datasets in a dataset root directory `[DATA_DIR]` as follows:

```
[DATA_DIR]
  │
  └── imagenet
  │   │
  │   └── train
  │   │
  │   └── val
  │  
  └── ImageNetV2-matched-frequency
  │   │
  │   └── 0
  │   │
  │   └── 1
  │   │
  │   └── ...
  │  
  └── imagenet-r
  │   │
  │   └── n01443537
  │   │
  │   └── n01484850
  │   │
  │   └── ...
  │
  └── objectnet-1.0
  │   │
  │   └── images
  │   │
  │   └── mappings
  │   │
  │   └── ...
  │
  └── sketch
  │   │
  │   └── n01440764
  │   │
  │   └── n01443537
  │   │
  │   └── ...
  │
  └── imagenet-a
      │
      └── n01498041
      │
      └── n01531178
      │
      └── ...
```

Note that the parameter `--data-dir` in all sample commands in `README.md` should be set to the dataset root directory, namely `[DATA_DIR]`.
