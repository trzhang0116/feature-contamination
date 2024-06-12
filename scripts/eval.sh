#!/bin/bash
python main.py \
--eval \
--data-dir [DATA_DIR] \
--eval-datasets ImageNet,ImageNetV2,ImageNetR,ObjectNet,ImageNetSketch,ImageNetA \
--exp-name eval_distill_clip_vitb16 \
--model-name CLIP_ViT-B/16 \
--load-dir [LOAD_DIR] \
--classifier-load-dir [CLASSIFIER_LOAD_DIR] \
--batch-size 512