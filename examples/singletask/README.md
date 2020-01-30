# Single-Task Training Example

In this example, we are going to train DeepLab-v3+ with the Xception-65 backbone for the task of semantic segmentation on NYUDv2.

## Prepare Data

Considering that you successfully installed the `DenseTorch` package, the next step is to download the NYUDv2 dataset with segmentation and depth masks. The dataset can be downloaded by following the [link](https://cloudstor.aarnet.edu.au/plus/s/XJKtmOKcTEnANZt). 

After downloading and unpacking the archive, create the `datasets` folder and link the `nyudv2` directory in the archive
to the just created folder:

```
mkdir datasets
ln -s /path/to/nyudv2 ./datasets/
```

## Training

Now you are ready to run the example script. To do so, simply execute `python train.py`. After it is finished, the
best model will be stored in the corresponding `pth.tar` file.

