# DenseTorch: PyTorch Wrapper for Smooth Workflow with Dense Per-Pixel Tasks

[![Build Status](https://api.travis-ci.com/DrSleep/DenseTorch.svg?branch=master)](https://travis-ci.com/DrSleep/DenseTorch)
[![Docs Status](https://readthedocs.org/projects/densetorch/badge/?version=latest)](https://densetorch.readthedocs.io/en/latest/)


This library aims to ease typical workflows involving dense per-pixel tasks in PyTorch. The progress in such tasks as semantic image segmentation and depth estimation have been significant over the last years, and in this library we provide an easy-to-setup environment for experimenting with given (or your own) models that reliably solve these tasks.

## Installation

Python >= 3.6.7 is supported.

```
git clone https://github.com/drsleep/densetorch.git
cd densetorch
pip install -e .
```

## Examples

Currently, we provide several models for single-task and multi-task setups:
 - `resnet` ResNet-18/34/50/101/152.
 - `mobilenet-v2` MobileNet-v2.
 - `xception-65` Xception-65.
 - `deeplab-v3+` DeepLab-v3+.
 - `lwrf` Light-Weight RefineNet.
 - `mtlwrf` Multi-Task Light-Weight RefineNet.

Examples are given in the `examples/` directory. Note that the provided examples do not necessarily reproduce the results achieved in corresponding papers, rather their goal is to demonstrate what can be done using this library.

## Motivation behind the library

As my everyday research is concerned with dense per-pixel tasks, I found myself oftentimes re-writing and updating (occassionally improving upon) my own code for each project. With the number of projects being on the rise recently, such an approach was no longer easy to manage. Hence, I decided to create a simple to use and simple to extend upon backbone (pun is not intended) structure, which I would be able to share with the community and, hopefully, ease the experience for others in the field.  

## Future Work

This library is still work-in-progress. More examples and more models will be added.
Contributions are welcome.

## Documentation

Is available [here](https://densetorch.readthedocs.io/en/latest/).

## Citation

If you found this library useful in your research, please consider citing
```
@misc{Nekrasov19,
  author = {Nekrasov, Vladimir},
  title = {DenseTorch},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/drsleep/densetorch}}
}
```
