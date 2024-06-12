#!/bin/bash
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