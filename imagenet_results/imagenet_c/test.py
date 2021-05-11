"""
Code modified from here: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
"""


from torch.autograd import Variable as V
from tqdm import tqdm

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch
import timm
import numpy as np
import argparse
import time
import os


parser = argparse.ArgumentParser(
    description="Evaluates robustness of various nets on ImageNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# Architecture
parser.add_argument(
    "--model-name",
    "-m",
    type=str,
    choices=[
        "resnetv2_50x1_bitm",
        "resnetv2_50x3_bitm",
        "resnetv2_101x1_bitm",
        "resnetv2_101x3_bitm",
        "resnetv2_152x4_bitm",
        "vit_base_patch16_224",
        "vit_base_patch32_224",
        "vit_large_patch16_224",
        "vit_large_patch32_224",
    ],
)
# Acceleration
parser.add_argument("--ngpu", type=int, default=4, help="0 = CPU.")
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////

if "bitm" in args.model_name:
    with torch.no_grad():
        net = timm.create_model(args.model_name, pretrained=True)
    args.test_bs = 4 * args.ngpu

elif "vit" in args.model_name:
    with torch.no_grad():
        net = timm.create_model(args.model_name, pretrained=True)
    args.test_bs = 8 * args.ngpu

args.prefetch = os.cpu_count()


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(1)
np.random.seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

print("Model Loaded")

# /////////////// Data Loader ///////////////

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
#     root="/home/jupyter/val",
#     transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
#     batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////


def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


# correct = 0
# for batch_idx, (data, target) in enumerate(clean_loader):
#     data = V(data.cuda(), volatile=True)
#
#     output = net(data)
#
#     pred = output.data.max(1)[1]
#     correct += pred.eq(target.cuda()).sum()
#
# clean_error = 1 - correct / len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []

    for severity in range(1, 6):
        distorted_dataset = dset.ImageFolder(
            root="imagenet2012_corrupted/" + distortion_name + "/" + str(severity),
            transform=trn.Compose(
                [trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]
            ),
        )

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset,
            batch_size=args.test_bs,
            shuffle=False,
            num_workers=args.prefetch,
            pin_memory=True,
        )

        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            with torch.no_grad():
                data = data.cuda()
                output = net(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()
        err = 1 - 1.0 * correct / len(distorted_dataset)
        errs.append(err.cpu().numpy())

    print("\n=Average", tuple(errs))
    return np.mean(errs)


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
print("\nUsing ImageNet data")

distortions = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

error_rates = []
for distortion_name in distortions:
    print(f"Currently evaluating: {distortion_name}")
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print(
        "Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}".format(
            distortion_name, 100 * rate
        )
    )


print(
    "mCE (unnormalized by AlexNet errors) (%): {:.2f}".format(
        100 * np.mean(error_rates)
    )
)
