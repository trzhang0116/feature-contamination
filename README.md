# Feature Contamination

PyTorch code accompanying the ICML'24 paper:

 [Feature Contamination: Neural Networks Learn Uncorrelated Features and Fail to Generalize](https://arxiv.org/pdf/2406.03345)

**TL;DR:** We identify that neural networks can learn "useless features" that are _uncorrelated_ with the label while learning useful features and, due to such a proclivity, fail to generalize under distribution shifts.

---

## Code for representation distillation

We provide code for representation distillation and linear probing on ImageNet and code for evaluating linear probes on ImageNet distribution shift datasets (ImageNet V2, ImageNet-R, ObjectNet, ImageNet Sketch, and ImageNet-A).


### Dependencies

```
python 3.10
pytorch >= 2.0
torchvision
clip
tqdm
```

To install all dependencies in a new Anaconda environment named `fc` (requires `CUDA >= 12.0`), run

```
conda env create
```

You can then activate the environment by

```
conda activate fc
```

### Data download

Please refer to `datasets.md` for instructions on downloading and organizing data.


### Training

Sample command for distilling CLIP ResNet-50 on ImageNet:

```
python main.py \
  --data-dir [DATA_DIR] \
  --exp-name distill_clip_rn50 \
  --model-name CLIP_RN50 \
  --rd \
  --distill-weight 1.0 \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --batch-size 256 \
  --distributed \
  --world-size 1 \
  --rank 0 \
  --dist-url tcp://127.0.0.1:29501 \
  --epochs 50 \
  --warmup-length 10000
```

Set the `--data-dir` parameter to your own dataset root directory. Please refer to `scripts/` for more sample commands for distilling other models (CLIP ResNet-101, ResNet-50x4, ResNet-50x16, and ViT-B/16).


### Linear probing

#### CLIP

Sample command for training a linear probe on CLIP ResNet-50:
```
python main.py \
  --data-dir [DATA_DIR] \
  --eval-datasets ImageNet,ImageNetV2,ImageNetR,ObjectNet,ImageNetSketch,ImageNetA \
  --exp-name lp_clip_rn50 \
  --model-name CLIP_RN50 \
  --lr 1e-3 \
  --lp \
  --weight-decay 0.1 \
  --batch-size 256 \
  --epochs 10 \
  --warmup-length 500
```

Set the `--data-dir` parameter to your own dataset root directory. You can change the `--model-name` parameter to use other CLIP models.

#### Distilled

Sample command for training a linear probe on a distilled CLIP ResNet-50:

```
python main.py \
  --data-dir [DATA_DIR] \
  --eval-datasets ImageNet,ImageNetV2,ImageNetR,ObjectNet,ImageNetSketch,ImageNetA \
  --exp-name lp_distill_clip_rn50 \
  --model-name CLIP_RN50 \
  --load-dir [LOAD_DIR] \
  --lr 1e-3 \
  --lp \
  --weight-decay 1e-3 \
  --batch-size 256 \
  --epochs 10 \
  --warmup-length 500
```

Set the `--data-dir` parameter to your own dataset root directory and set the `--load-dir` parameter to the path to a distilled model (should end with '.pt', '.pth', etc.).


### Evaluating trained linear probes

Sample command for evaluating a linear probe trained on a distilled CLIP RN-50:

```
python main.py \
  --eval \
  --data-dir [DATA_DIR] \
  --eval-datasets ImageNet,ImageNetV2,ImageNetR,ObjectNet,ImageNetSketch,ImageNetA \
  --exp-name eval_distill_clip_rn50 \
  --model-name CLIP_RN50 \
  --load-dir [LOAD_DIR] \
  --classifier-load-dir [CLASSIFIER_LOAD_DIR] \
  --batch-size 512
```

Set the `--data-dir` parameter to your own dataset root directory, set the `--load-dir` parameter to the path to a distilled model (should end with '.pt', '.pth', etc.), and set the `--classifier-load-dir` parameter to the path to a linear probe (should end with '.pt', '.pth', etc.).

### Pre-trained models

Distilled CLIP ResNet-50, ResNet-101, ResNet-50x4, ResNet-50x16, and ViT-B/16 models and the corresponding linear probes can be found in [this Google Drive](https://drive.google.com/drive/folders/1FNs-gPvr7_xYizLV44oN2i21mo1hlpRL?usp=drive_link
).

---

## Code for synthetic data experiments

We provide code for reproducing our numerical results in the paper in `toy.py`. Sample command for running:

```
python toy.py
```

Data and model parameters are listed at the front of the Python file and you can change them to explore different configurations (e.g., loss functions, optimizers, activation functions, etc.). In particular, to reproduce the empirical results that match our theoretical setting, set `loss_name = "hinge"`, `opt="sgd"`, `nonlinear=False`, and `mlp_freeze_output=True`.

---

## Code for feature visualization on CIFAR-10

See `cifar10/` for details.

---

## Acknowledgements

The code structure of this repository is based on [wise-ft](https://github.com/mlfoundations/wise-ft). We greatly thank the authors for releasing their code!


## Citation

If you found this repository useful, please kindly cite:

```
@inproceedings{zhang2024feature,
  title={Feature Contamination: Neural Networks Learn Uncorrelated Features and Fail to Generalize},
  author={Zhang, Tianren and Zhao, Chujie and Chen, Guanyu and Jiang, Yizhou and Chen, Feng},
  booktitle={ICML},
  year={2024}
}
```
