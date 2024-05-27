import os

# Define the range of keep ratios you want to evaluate
keep_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Common command template
command_template = (
    "python eval_vit_ng.py "
    "--pretrain_ckpt /ppio_net0/pretrained/byol-imagenet32-t3pmk238-ep=999.ckpt "
    "--data_path /ppio_net0/torch_ds/imagenet "
    "--dataset imagenet "
    "--epochs 30 "
    "--lr 0.1 "
    "--batch_size 256 "
    "--num_gpus 1 "
    "--num_workers 8 "
    "--project NG-Transformer "
    "--run_name ViT-NG-ratio{ratio} "
    "--keep_ratio {ratio}"
)

# Iterate through the defined keep ratios and run the command for each
for ratio in keep_ratios:
    command = command_template.format(ratio=ratio)
    os.system(command)
