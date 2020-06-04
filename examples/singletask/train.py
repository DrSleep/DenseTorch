import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import densetorch as dt
from config import *

# model_options
## encoder
pretrained = True
return_idx = [1, 20]
config = 8  # output stride 8

# set seeds
dt.misc.set_seed(seed)

# data setup
transform_common = [dt.data.Normalise(*normalise_params), dt.data.ToTensor()]
transform_train = transforms.Compose(
    [dt.data.RandomMirror(), dt.data.RandomCrop(crop_size)] + transform_common
)
transform_val = transforms.Compose(transform_common)
trainloader = DataLoader(
    dt.data.MMDataset(
        data_file, data_dir, line_to_paths_fn, masks_names, transform=transform_train,
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
valloader = DataLoader(
    dt.data.MMDataset(
        val_file, data_val_dir, line_to_paths_fn, masks_names, transform=transform_val,
    ),
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

# model setup
enc = dt.nn.xception65(pretrained=pretrained, return_idx=return_idx)
dec = dt.nn.DLv3plus(enc._out_c, num_classes, rates=enc.rates)
model1 = nn.DataParallel(nn.Sequential(enc, dec).cuda())
print("Model has {} parameters".format(dt.misc.compute_params(model1)))

# optim setup
optims = [
    dt.misc.create_optim(
        optim_enc, enc.parameters(), lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc
    ),
    dt.misc.create_optim(
        optim_dec, dec.parameters(), lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec
    ),
]

# schedulers
opt_scheds = []
for opt in optims:
    opt_scheds.append(
        torch.optim.lr_scheduler.MultiStepLR(
            opt, np.arange(1, n_epochs, 100), gamma=0.1
        )
    )

for i in range(n_epochs):
    for sched in opt_scheds:
        sched.step(i)
    model1.train()
    print("Epoch {:d}".format(i))
    dt.engine.train(model1, optims, [crit_segm], trainloader, loss_coeffs)
    if i % val_every == 0:
        metrics = [dt.engine.MeanIoU(num_classes)]
        model1.eval()
        with torch.no_grad():
            vals = dt.engine.validate(model1, metrics, valloader)
        saver.save(
            new_val=vals, dict_to_save={"state_dict": model1.state_dict(), "epoch": i}
        )
