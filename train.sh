python train_linear.py \
      --pretrain_ckpt /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --data_path /ppio_net0/torch_ds \
      --dataset cifar100 \
      --epochs 30 \
      --lr 0.1 \
      --batch_size 256 \
      --project NG-Transformer \
      --run_name Linear \
      --num_gpus 1
