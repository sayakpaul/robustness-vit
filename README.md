# Vision Transformers are Robust Learners

This repository contains the code for the paper [Vision Transformers are Robust Learners] (Link to be updated) by Sayak Paul and Pin-Yu Chen.

_**Abstract**_

TBA

## Structure and Navigation

All the results related to the ImageNet datasets (ImageNet-C, ImageNet-P, ImageNet-R, ImageNet-A, ImageNet-O, and ImageNet-9) can be derived from the notebooks contained in the [`imagenet_results`](https://github.com/sayakpaul/robustness-vit/tree/master/imagenet_results) directory. Many notebooks inside that directory can be executed with [Google Colab](https://colab.research.google.com/). When that is not the case, we provide execution instructions explicitly. This is followed for the rest of the directories present inside this repository. 

[`analysis`](https://github.com/sayakpaul/robustness-vit/tree/master/analysis) directory contains the code used to generate results for Section 4 in the paper. 

[`misc`](https://github.com/sayakpaul/robustness-vit/tree/master/misc) directory contains the code for visualizing frequency artifacts inside images. 

## About our dev environment

We use Python 3.8. As for the hardware setup (when not using Colab), we use a [GCP AI Platform Notebook](https://cloud.google.com/ai-platform-notebooks) with 4 V100s, 60 GBs of RAM with 16 vCPUs (`n1-standard-16` [machine type](https://cloud.google.com/compute/docs/machine-types)).

## Citation

TBA

## Acknowledgements

TBA
