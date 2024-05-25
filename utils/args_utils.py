import argparse


def parse_args() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="NG-Transformer")
    parser.add_argument("--entity", type=str, default="pigpeppa")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_path", type=str, default="/ppio_net0/torch_ds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_gpus", type=int, default=1)

    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--dim_feature", type=int, default=2048)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pretrain_ckpt", type=str, required=True)

    # parse args
    args = parser.parse_args()
    return args
