For this directory, a local setup is needed as well as you need to download the ImageNet-1k validation set. The structure should resemble what's specified in here - [`imagenet_results/imagenet_o/README.md`] (Link to be updated).

We used an [open-source implementation](https://github.com/gatheluck/FourierHeatmap) of the paper [A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/abs/1906.08988) for getting our results. So, after getting the ImageNet-1k validation set and structuring it, simply follow the instructions from that respository. For loading the BiT (m r101x3) and ViT (L-16) models with pre-trained weights we used `timm` which can be installed with `pip install timm`. The `mean` and `std` parameters should be `[0.5, 0.5, 0.5]`.

We randomly sampled 1000 images from the ImageNet-1k validation set for this experiment. The exact imagepaths can be downloaded using:

```shell
$ gdown --id 1QtAJsTjBOf3CnrTzTTqP-nPnHcTc2g9E
```

[`gdown`](https://pypi.org/project/gdown/) is a simple tool letting us download public Google Drive files in a seamless manner. Install it via - `pip install gdown`. Also, note that this is a resource-intensive experiment.

We thank Justin Gilmer for helpful discussions. Justin is one of the authors of the paper ([A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/abs/1906.08988)) on which this experiment is based. 