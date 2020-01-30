# Multi-Task Training Example

In this example, we are going to train Multi-Task Light-Weight RefineNet for joint semantic segmentation and depth estimation. Note that inference examples together with pre-trained weights can be found in the official [repository](https://github.com/DrSleep/multi-task-refinenet).

The hyperparameters set here are not the same as used in the corresponding paper, hence the results will differ. Please refer to the paper below for more information on the
model and the training regime.

```
Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations
Vladimir Nekrasov, Thanuja Dharmasiri, Andrew Spek, Tom Drummond, Chunhua Shen, Ian Reid
In ICRA 2019
```

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
best model will be stored in the corresponding `pth.tar` file. Note that it would the model that improves upon the
previous checkpoint both in terms of mean IoU (for segmentation) and linear RMSE (for depth estimation).


