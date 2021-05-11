This directory assumes you have already downloaded the **ImageNet-C** dataset and have installed the Python libraries specified in `requirements.txt`. If you haven't already then check out the download instructions [here](https://github.com/hendrycks/robustness#imagenet-c). We kept the ImageNet-C dataset under `imagenet2012_corrupted` directory and specified it accordingly during recording the scores. To record the unnormalized corruption errors (CE), we used (and modified) the [original script](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py) which is named as `test.py`.

To obtain the unnormalized CEs for, say, ViT L-16, simply run the following from a terminal:

```shell
$ python test.py -m vit_large_patch16_224
```

You may need to adjust the batch size and the number of GPUs according to the available infrastructure. 

We thank Dan Hendrycks for helpful discussions. Dan is the main author of the ImageNet-C dataset.