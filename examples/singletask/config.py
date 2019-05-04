import operator

import numpy as np
import torch.nn as nn

import densetorch as dt

# Random seed
seed = 42

# Data settings
crop_size = 450
batch_size = 4
val_batch_size = 4
num_classes = 40
n_epochs = 1000
val_every = 5

data_file = './lists/train_list_depth.txt'
val_file = './lists/val_list_depth.txt'
data_dir = './datasets/nyudv2/'
data_val_dir = './datasets/nyudv2/'
masks_names = ('segm',)
def line_to_paths_fn(x):
    rgb, segm, depth = x.decode('utf-8').strip('\n').split('\t')
    return [rgb, segm]
depth_scale = 5000.
img_scale = 1. / 255
img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
normalise_params = [img_scale, # SCALE
                    img_mean.reshape((1, 1, 3)), # MEAN
                    img_std.reshape((1, 1, 3)),
                    depth_scale] # STD
ignore_index = 255

# optim options
crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()

lr_enc = 1e-2
optim_enc = 'SGD'
mom_enc = 0.9
wd_enc = 1e-4
lr_dec = 5e-2
optim_dec = 'SGD'
mom_dec = 0.9
wd_dec = 1e-4
loss_coeffs = (1.,)

# saving criterions
init_vals = (0.,)
comp_fns = [operator.gt]
saver = dt.misc.Saver(init_vals, comp_fns)
