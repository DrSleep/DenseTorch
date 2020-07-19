import argparse
import json

from densetorch.misc import broadcast


def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(
        description="Arguments for DenseTorch Training Pipeline"
    )

    # Common transformations
    parser.add_argument("--img-scale", type=float, default=1.0 / 255)
    parser.add_argument(
        "--img-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406)
    )
    parser.add_argument("--img-std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    parser.add_argument("--depth-scale", type=float, default=5000.0)

    # Training augmentations
    parser.add_argument(
        "--augmentations-type",
        type=str,
        choices=["densetorch", "albumentations"],
        default="densetorch",
    )

    # Dataset
    parser.add_argument(
        "--val-list-path", type=str, default="./data/val.nyu",
    )
    parser.add_argument(
        "--val-dir", type=str, default="./datasets/nyud/",
    )
    parser.add_argument("--val-batch-size", type=int, default=1)

    # Optimisation
    parser.add_argument(
        "--enc-optim-type", type=str, default="sgd",
    )
    parser.add_argument(
        "--dec-optim-type", type=str, default="sgd",
    )
    parser.add_argument(
        "--enc-lr", type=float, default=5e-4,
    )
    parser.add_argument(
        "--dec-lr", type=float, default=5e-3,
    )
    parser.add_argument(
        "--enc-weight-decay", type=float, default=1e-5,
    )
    parser.add_argument(
        "--dec-weight-decay", type=float, default=1e-5,
    )
    parser.add_argument(
        "--enc-momentum", type=float, default=0.9,
    )
    parser.add_argument(
        "--dec-momentum", type=float, default=0.9,
    )
    parser.add_argument(
        "--enc-lr-gamma",
        type=float,
        default=0.5,
        help="Multilpy lr_enc by this value after each stage.",
    )
    parser.add_argument(
        "--dec-lr-gamma",
        type=float,
        default=0.5,
        help="Multilpy lr_dec by this value after each stage.",
    )
    parser.add_argument(
        "--enc-scheduler-type",
        type=str,
        choices=["poly", "multistep"],
        default="multistep",
    )
    parser.add_argument(
        "--dec-scheduler-type",
        type=str,
        choices=["poly", "multistep"],
        default="multistep",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Ignore this label in the training loss.",
    )
    parser.add_argument("--random-seed", type=int, default=42)

    # Architecture setup
    parser.add_argument(
        "--enc-backbone",
        type=str,
        choices=[
            "xception65",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "mobilenetv2",
        ],
        default="resnet50",
    )
    parser.add_argument("--enc-pretrained", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--enc-return-layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Indices of layers to return from the encoder. Usually, each index corresponds to a layer with a different resolution.",
    )
    parser.add_argument(
        "--dec-backbone",
        type=str,
        choices=["dlv3plus", "lwrefinenet"],
        default="lwrefinenet",
    )
    parser.add_argument(
        "--dec-combine-layers",
        type=str,
        default="[0, 1, 2, 3]",
        help="Comma-separated list of (lists of) indices of input layers to combine via element-wise summation in the decoder. Assumes that the corresponding layers are of the same size.",
    )

    # Tasks setup
    parser.add_argument("--num-classes", type=int, nargs="+", default=[40])
    parser.add_argument(
        "--tasks",
        type=str,
        choices=["segm", "depth", "normals"],
        default=["segm"],
        nargs="+",
    )
    parser.add_argument("--tasks-loss-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--ignore-indices", type=int, default=[255], nargs="+")
    parser.add_argument(
        "--initial-values",
        type=float,
        default=[0.0],
        nargs="+",
        help="Initial values for each task",
    )
    parser.add_argument(
        "--saving-criterions",
        type=str,
        choices=["up", "down"],
        default=["up"],
        nargs="+",
        help="Save the checkpoint when the new metric value has either gone `up` or `down`",
    )

    # Training / validation setup
    parser.add_argument(
        "--data-list-sep",
        type=str,
        default="\t",
        help="Separator for val and train lists.",
    )
    parser.add_argument(
        "--data-list-columns",
        type=str,
        nargs="+",
        default=["image", "segm", "depth", "normals"],
        help="Name of columns in val and train lists. Must always have 'image'.",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=3,
        help="Number of training stages. All other arguments with nargs='+' must "
        "have the number of arguments equal to this value. Otherwise, the given "
        "arguments will be broadcasted to have the required length.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="densetorch",
        choices=["densetorch", "torchvision"],
    )
    parser.add_argument(
        "--val-download",
        type=int,
        choices=[0, 1],
        default=0,
        help="Only used if dataset_type == torchvision.",
    )

    # Checkpointing configuration
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./checkpoints/checkpoint.pth.tar",
        help="Path to the checkpoint file.",
    )

    # Arguments broadcastable across training stages
    stage_parser = parser.add_argument_group("stage-parser")
    stage_parser.add_argument(
        "--crop-size", type=int, nargs="+", default=(500, 500, 500,)
    )
    stage_parser.add_argument(
        "--shorter-side", type=int, nargs="+", default=(350, 350, 350,)
    )
    stage_parser.add_argument(
        "--low-scale", type=float, nargs="+", default=(0.5, 0.5, 0.5,)
    )
    stage_parser.add_argument(
        "--high-scale", type=float, nargs="+", default=(2.0, 2.0, 2.0,)
    )
    stage_parser.add_argument(
        "--train-list-path", type=str, nargs="+", default=("./data/train.nyu",)
    )
    stage_parser.add_argument(
        "--train-dir", type=str, nargs="+", default=("./datasets/nyud/",)
    )
    stage_parser.add_argument(
        "--train-batch-size", type=int, nargs="+", default=(6, 6, 6,)
    )
    stage_parser.add_argument(
        "--freeze-bn", type=int, choices=[0, 1], nargs="+", default=(1, 1, 1,)
    )
    stage_parser.add_argument(
        "--epochs-per-stage", type=int, nargs="+", default=(100, 100, 100),
    )
    stage_parser.add_argument("--val-every", type=int, nargs="+", default=(5, 5, 5,))
    stage_parser.add_argument(
        "--stage-names",
        type=str,
        nargs="+",
        choices=["SBD", "VOC"],
        default=("SBD", "VOC",),
        help="Only used if dataset_type == torchvision.",
    )
    stage_parser.add_argument(
        "--train-download",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=(0, 0,),
        help="Only used if dataset_type == torchvision.",
    )
    stage_parser.add_argument(
        "--grad-norm",
        type=float,
        nargs="+",
        default=(0.0,),
        help="If > 0.0, clip gradients' norm to this value.",
    )
    args = parser.parse_args()
    # Broadcast all arguments in stage-parser
    for group_action in stage_parser._group_actions:
        argument_name = group_action.dest
        setattr(
            args,
            argument_name,
            broadcast(getattr(args, argument_name), args.num_stages),
        )
    # TODO: parse errors for combination of encoder / decoder return layers and combine layers
    if len(args.num_classes) != len(args.tasks) or len(args.tasks) != len(
        args.ignore_indices
    ):
        parser.error(
            f"Length of num_classes, tasks and ignore indices must be the same, "
            f"Got {len(args.num_classes):d}, {len(args.tasks):d} and {len(args.ignore_indices):d}"
        )
    if "image" not in args.data_list_columns:
        parser.error(
            f"data_list_columns must have the `image` entry, received {args.data_list_columns}"
        )
    if set(args.tasks) - set(args.data_list_columns) != set():
        parser.error(
            f"Tasks not present in `data_list_columns` were present: {set(args.tasks) - set(args.data_list_columns)}."
        )
    try:
        args.dec_combine_layers = json.loads(args.dec_combine_layers)
    except Exception as e:
        parser.error(f"Cannot parse `dec_combine_layers`: {e.message}")
    return args
