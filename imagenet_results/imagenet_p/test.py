"""
Code modified from here: https://github.com/hendrycks/robustness/blob/master/ImageNet-P/test.py
"""

from scipy.stats import rankdata
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trn_F
import torchvision.models as models
import numpy as np
import argparse
import torch
import timm
import os

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.video_loader import VideoFolder

parser = argparse.ArgumentParser(
    description="Evaluates robustness of various nets on ImageNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# Architecture
parser.add_argument(
    "--model-name",
    "-m",
    required=True,
    type=str,
    choices=["resnetv2_101x3_bitm", "vit_large_patch16_224"],
)
parser.add_argument(
    "--perturbation",
    "-p",
    default="brightness",
    type=str,
    choices=[
        "gaussian_noise",
        "shot_noise",
        "motion_blur",
        "zoom_blur",
        "spatter",
        "brightness",
        "translate",
        "rotate",
        "tilt",
        "scale",
        "speckle_noise",
        "gaussian_blur",
        "snow",
        "shear",
    ] # Only 10 of these perturbations are available
)
parser.add_argument("--difficulty", "-d", type=int, default=1, choices=[1, 2, 3])
# Acceleration
parser.add_argument("--ngpu", type=int, default=4, help="0 = CPU.")
args = parser.parse_args()
print(args)

# /////////////// Norm Stats ///////////////
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# /////////////// Model Setup //////////////

if args.model_name == "resnetv2_101x3_bitm":
    net = timm.create_model("resnetv2_101x3_bitm", pretrained=True)
    args.test_bs = 3 * 4

elif args.model_name == "vit_large_patch16_224":
    net = timm.create_model("vit_large_patch16_224", pretrained=True)
    args.test_bs = 4 * 4

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

print("Model Loaded\n")

############# Data loader #############

transforms = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.difficulty > 1 and "noise" in args.perturbation:
    loader = torch.utils.data.DataLoader(
        VideoFolder(
            root="ImageNet-P/" + args.perturbation + "_" + str(args.difficulty),
            transform=transforms,
        ),
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
else:
    loader = torch.utils.data.DataLoader(
        VideoFolder(root="ImageNet-P/" + args.perturbation, transform=transforms),
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

print("Data Loaded\n")


# /////////////// Stability Measurements ///////////////

identity = np.asarray(range(1, 1001))
cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (999 - 5)))
recip = 1.0 / identity

# def top5_dist(sigma):
#     result = 0
#     for i in range(1,6):
#         for j in range(min(sigma[i-1], i) + 1, max(sigma[i-1], i) + 1):
#             if 1 <= j - 1 <= 5:
#                 result += 1
#     return result


def dist(sigma, mode="top5"):
    if mode == "top5":
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma - 1][:5]))
    elif mode == "zipf":
        return np.sum(np.abs(recip - recip[sigma - 1]) * recip)


def ranking_dist(
    ranks,
    noise_perturbation=True if "noise" in args.perturbation else False,
    mode="top5",
):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(
    predictions, noise_perturbation=True if "noise" in args.perturbation else False
):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation:
                    prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


# /////////////// Get Results ///////////////

from tqdm import tqdm

predictions, ranks = [], []
with torch.no_grad():

    for data, target in loader:
        num_vids = data.size(0)
        data = data.view(-1, 3, 224, 224).cuda()

        output = net(data)

        for vid in output.view(num_vids, -1, 1000):
            predictions.append(vid.argmax(1).to("cpu").numpy())
            ranks.append(
                [
                    np.uint16(rankdata(-frame, method="ordinal"))
                    for frame in vid.to("cpu").numpy()
                ]
            )


ranks = np.asarray(ranks)

print("Computing Metrics\n")

print("Flipping Prob\t{:.5f}".format(flip_prob(predictions)))
print("Top5 Distance\t{:.5f}".format(ranking_dist(ranks, mode="top5")))
print("Zipf Distance\t{:.5f}".format(ranking_dist(ranks, mode="zipf")))
