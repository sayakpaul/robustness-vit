This directory requires the ImageNet-1k validation set and the ImageNet-O dataset to be present under `val` and `imagenet-o` directories respectively. Download instructions for ImageNet-1k are available [here](https://image-net.org/download.php) and the same for ImageNet-O are [here](https://github.com/hendrycks/natural-adv-examples). Here we provide two Jupyter Notebooks to assess a family of BiT and ViT models. Hence, the Jupyter Notebooks has been named accordingly. For calculating, Area Under Precision-Recall (AUPR) we used (and modified) the [official script](https://github.com/hendrycks/natural-adv-examples/blob/master/calibration_tools.py). 

Python library requirements are specified in `requirements.txt` and further execution instructions can be found in the notebooks. To install a GPU-compatible version of JAX, follow the instructions from [here](https://github.com/google/jax#pip-installation). 

Before running the ViT notebook, please clone the `vision_transformer` repository first:

```shell
$ git clone --depth 1 https://github.com/google-research/vision_transformer
```

Inside the ImageNet-1k validation set the images should be organized with respect to their respective categories so that a particular category contains all the images under it.  
