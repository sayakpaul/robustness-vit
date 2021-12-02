# Vision Transformers are Robust Learners

This repository contains the code for the paper [Vision Transformers are Robust Learners](https://arxiv.org/abs/2105.07581) by Sayak Paul<sup>\*</sup> and Pin-Yu Chen<sup>\*</sup>.

<sup>\*</sup>Equal contribution.

### Abstract

Transformers, composed of multiple self-attention layers, hold strong promises toward a generic learning primitive applicable to different data modalities, including the recent breakthroughs in computer vision achieving state-of-the-art (SOTA) standard accuracy with better parameter efficiency. Since self-attention helps a model systematically align different components present inside the input data, it leaves grounds to investigate its performance under model robustness benchmarks. In this work, we study the robustness of the Vision Transformer (ViT) against common corruptions and perturbations, distribution shifts, and natural adversarial examples. We use six different diverse ImageNet datasets concerning robust classification to conduct a comprehensive performance comparison of ViT models and SOTA convolutional neural networks (CNNs), Big-Transfer. Through a series of six systematically designed experiments, we then present analyses that provide both quantitative and qualitative indications to explain why ViTs are indeed more robust learners. For example, with fewer parameters and similar dataset and pre-training combinations, ViT gives a top-1 accuracy of 28.10% on ImageNet-A which is 4.3x higher than a comparable variant of BiT. Our analyses on image masking, Fourier spectrum sensitivity, and spread on discrete cosine energy spectrum reveal intriguing properties of ViT attributing to improved robustness. 

## Structure and Navigation

All the results related to the ImageNet datasets (ImageNet-C, ImageNet-P, ImageNet-R, ImageNet-A, ImageNet-O, and ImageNet-9) can be derived from the notebooks contained in the [`imagenet_results/`](https://github.com/sayakpaul/robustness-vit/tree/master/imagenet_results) directory. Many notebooks inside that directory can be executed with [Google Colab](https://colab.research.google.com/). When that is not the case, we provide execution instructions explicitly. This is followed for the rest of the directories present inside this repository. 

[`analysis/`](https://github.com/sayakpaul/robustness-vit/tree/master/analysis) directory contains the code used to generate results for Section 4 in the paper. 

[`misc/`](https://github.com/sayakpaul/robustness-vit/tree/master/misc) directory contains the code for visualizing frequency artifacts inside images. 

For any questions, please open an issue and tag @sayakpaul.

## About our dev environment

We use Python 3.8. As for the hardware setup (when not using Colab), we use [GCP Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench) with
4 V100s, 60 GBs of RAM with 16 vCPUs (`n1-standard-16` [machine type](https://cloud.google.com/compute/docs/machine-types)).

## Citation

```
@article{paul2021vision,
  title={Vision Transformers are Robust Learners},
  author={Sayak Paul and Pin-Yu Chen},
  journal={AAAI},
  year={2022}
}
```

## Acknowledgements

We are thankful to the [Google Developers Experts program](https://developers.google.com/programs/experts/) (specifically Soonson Kwon and Karl Weinmeister) for providing Google Cloud Platform credits to support the experiments. We also thank Justin Gilmer (of Google), Guillermo Ortiz-Jimenez (of EPFL, Switzerland), and Dan Hendrycks (of UC Berkeley) for fruitful discussions.
