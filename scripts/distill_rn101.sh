#!/bin/bash
python main.py \
--data-dir [DATA_DIR] \
--exp-name distill_clip_rn101 \
--model-name CLIP_RN101 \
--rd \
--lr 1e-3 \
--weight-decay 0.1 \
--batch-size 256 \
--distributed \
--world-size 1 \
--rank 0 \
--dist-url tcp://127.0.0.1:29501 \
--epochs 50 \
--warmup-length 10000