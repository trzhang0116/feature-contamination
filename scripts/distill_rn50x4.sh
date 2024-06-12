#!/bin/bash
python main.py \
--data-dir [DATA_DIR] \
--exp-name distill_clip_rn50x4 \
--model-name CLIP_RN50x4 \
--rd \
--lr 1e-4 \
--weight-decay 0.5 \
--batch-size 256 \
--distributed \
--world-size 1 \
--rank 0 \
--dist-url tcp://127.0.0.1:29501 \
--epochs 50 \
--warmup-length 10000