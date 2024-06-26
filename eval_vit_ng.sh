
python eval_vit_ng.py \
      --pretrain_ckpt /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --data_path /ppio_net0/torch_ds/imagenet \
      --dataset imagenet \
      --epochs 30 \
      --lr 0.1 \
      --batch_size 256 \
      --num_gpus 1 \
      --num_workers 8 \
      --project NG-Transformer \
      --run_name ViT-NG

