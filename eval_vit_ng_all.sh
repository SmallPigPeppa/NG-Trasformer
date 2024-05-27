#!/bin/bash

PRETRAIN_CKPT="/ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt"
DATA_PATH="/ppio_net0/torch_ds/imagenet"
DATASET="imagenet"
EPOCHS=30
LR=0.1
BATCH_SIZE=256
NUM_GPUS=1
NUM_WORKERS=8
PROJECT="NG-Transformer"

for RATIO in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
do
  RUN_NAME="ViT-NG-ratio$RATIO"
  echo "Running evaluation with --keep_ratio $RATIO"
  python eval_vit_ng.py \
      --pretrain_ckpt $PRETRAIN_CKPT \
      --data_path $DATA_PATH \
      --dataset $DATASET \
      --epochs $EPOCHS \
      --lr $LR \
      --batch_size $BATCH_SIZE \
      --num_gpus $NUM_GPUS \
      --num_workers $NUM_WORKERS \
      --project $PROJECT \
      --run_name $RUN_NAME \
      --keep_ratio $RATIO
done
