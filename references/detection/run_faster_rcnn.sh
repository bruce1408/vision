#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 train.py \
--dataset coco \
--model fasterrcnn_resnet50_fpn \
--epochs 26 \
--batch-size 8 \
--lr-steps 16 22 \
--output-dir ./outputs \
--resume /home/bruce_ultra/workspace/vision-main/references/detection/outputs/checkpoint.pth \
--aspect-ratio-group-factor 3 \
--weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
--data-path /DataVault/datasets/coco2017
