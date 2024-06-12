#!/bin/bash
python main.py \
--data-dir [DATA_DIR] \
--exp-name distill_clip_vitb16 \
--model-name CLIP_ViT-B/16 \
--rd \
--lr 5e-5 \
--weight-decay 0.1 \
--batch-size 256 \
--distributed \
--world-size 1 \
--rank 0 \
--dist-url tcp://127.0.0.1:29502 \
--epochs 300 \
--print-freq 200 \
--warmup-length 10000