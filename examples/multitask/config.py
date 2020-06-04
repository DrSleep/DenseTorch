import operator

import numpy as np
import torch.nn as nn

import densetorch as dt

# Random seed
seed = 42

# Data settings
crop_size = 400
batch_size = 4
val_batch_size = 4
num_classes = (40, 1)
n_epochs = 1000
val_every = 5

data_file = "./lists/train_list_depth.txt"
val_file = "./lists/val_list_depth.txt"
data_dir = "./datasets/nyudv2/"
data_val_dir = "./datasets/nyudv2/"
masks_names = ("segm", "depth")


def line_to_paths_fn(x):
    return x.decode("utf-8").strip("\n").split("\t")


depth_scale = 5000.0
img_scale = 1.0 / 255
img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
normalise_params = [
    img_scale,  # SCALE
    img_mean.reshape((1, 1, 3)),  # MEAN
    img_std.reshape((1, 1, 3)),
    depth_scale,
]  # STD
ignore_index = 255
ignore_depth = 0

# optim options
crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
crit_depth = dt.engine.InvHuberLoss(ignore_index=ignore_depth).cuda()

lr_enc = 1e-3
optim_enc = "SGD"
mom_enc = 0.9
wd_enc = 1e-5
lr_dec = 1e-2
optim_dec = "SGD"
mom_dec = 0.9
wd_dec = 1e-5
loss_coeffs = (0.5, 0.5)  # equal weights per task

# saving criterions
init_vals = (0.0, 10000.0)
comp_fns = [operator.gt, operator.lt]
ckpt_dir = "./"
ckpt_path = "./checkpoint.pth.tar"
saver = dt.misc.Saver(
    args=locals(),
    ckpt_dir=ckpt_dir,
    best_val=init_vals,
    condition=comp_fns,
    save_several_mode=all,
)
