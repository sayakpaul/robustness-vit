This directory assumes you have already downloaded the **ImageNet-P** dataset and have installed the Python libraries specified in `requirements.txt`. If you haven't already then check out the download instructions [here](https://github.com/hendrycks/robustness#imagenet-p). We kept the ImageNet-C dataset under `ImageNet-P` directory and specified it accordingly during recording the scores. To record the unnormalized flip rate (FR) and top-5 distance (T5D), we used (and modified) the [original script](https://github.com/hendrycks/robustness/blob/master/ImageNet-P/test.py) which is named as `test.py`.

To obtain the unnormalized FR and T5D for, say, ViT L-16 and for the "Motion Blur" perturbation, simply run the following from a terminal:

```shell
$ python test.py -m vit_large_patch16_224 -p motion_blur
```

You may need to adjust the batch size and the number of GPUs according to the available infrastructure. Also, we always recorded the FRs for difficulty level 1. 

We thank Dan Hendrycks for helpful discussions. Dan is the main author of the ImageNet-C dataset.