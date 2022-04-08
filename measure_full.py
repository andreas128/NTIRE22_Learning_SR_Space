from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import tqdm
import sys
import os
import numpy as np
import pandas as pd
import scipy.signal

import cv2
from setuptools import glob

from imresize import imresize

if True:
    sys.path.insert(0, "./PerceptualSimilarity")
    from lpips import lpips


def fiFindByWildcard(wildcard):
    return glob.glob(os.path.expanduser(wildcard), recursive=True)


def dprint(d):
    out = []
    for k, v in d.items():
        out.append(f"{k}: {v:0.4f}")
    print(", ".join(out))


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img[:, :, [2, 1, 0]]


def lpips_analysis(gt, srs, scale):
    from collections import OrderedDict
    results = OrderedDict()

    gt_img = imread(gt)
    h, w, _ = gt_img.shape
    gt_img = gt_img[:(h//8)*8, :(w//8)*8]
    print("Compare GT", gt, gt_img.shape)

    sr_imgs = [imread(sr) for sr in srs]

    for sr_img, sr in zip(sr_imgs, srs):
        print("   with SR", sr, sr_img.shape)

    n_samples = len(sr_imgs)

    lpipses_sp = []
    lpipses_gl = []

    mses_sp = []
    mses_gl = []

    kernel = np.ones((16, 16)) / (16 * 16)

    for sample_idx in tqdm.tqdm(range(n_samples)):
        sr = sr_imgs[sample_idx]

        h1, w1, _ = gt_img.shape
        sr = sr[:h1, :w1]

        # LPIPS
        lpips_sp = loss_fn_alex_sp(2 * t(sr) - 1, 2 * t(gt_img) - 1)
        lpipses_sp.append(lpips_sp)
        lpipses_gl.append(lpips_sp.mean().item())

        # MSE
        mse_sp_rgb = ((sr * 1.0) - (gt_img * 1.0)) ** 2
        assert mse_sp_rgb.shape[2] == 3
        mse_sp = mse_sp_rgb.mean(axis=2)
        mse_sp = scipy.signal.convolve2d(mse_sp, kernel)
        mse_sp = torch.Tensor(mse_sp)

        mses_sp.append(mse_sp)
        mses_gl.append(mse_sp.mean())

    for n in range(1, n_samples):
        # LPIPS
        lpips_gl = np.min(lpipses_gl[: n])

        results[f'LPIPS_mean_n{n}'] = np.mean(lpipses_gl[: n])

        lpipses_stacked = torch.stack(
            [l[0, 0, :, :] for l in lpipses_sp[: n]], dim=2)

        lpips_best_sp, _ = torch.min(lpipses_stacked, dim=2)

        lpips_loc = lpips_best_sp.mean().item()
        results[f'LPIPS_best_loc_n{n}'] = lpips_loc

        lpips_diff = (lpips_gl - lpips_loc)
        results[f'LPIPS_diff_n{n}'] = lpips_diff

        score = lpips_diff / lpips_gl * 100

        results[f'LPIPS_score_n{n}'] = score

        # MSE
        mse_gl = np.min(mses_gl[: n])

        results[f'MSE_mean_n{n}'] = np.mean(mses_gl[: n])

        mses_stacked = torch.stack(mses_sp[: n], dim=2)

        mse_best_sp, _ = torch.min(mses_stacked, dim=2)

        mse_loc = mse_best_sp.mean().item()
        results[f'MSE_best_loc_n{n}'] = mse_loc

        mse_diff = (mse_gl - mse_loc)
        results[f'MSE_diff_n{n}'] = mse_diff

        score = mse_diff / mse_gl * 100

        results[f'MSE_score_n{n}'] = score

        dprint(results)

    return results


name, gt_dir, srs_dir, n_samples, scale, n_max = sys.argv[1:]

gt_dir = os.path.expanduser(gt_dir)
srs_dir = os.path.expanduser(srs_dir)
n_samples = int(n_samples)
n_max = int(n_max)
scale = int(scale)

########################################
# Get Paths
########################################

gt_imgs_raw = fiFindByWildcard(os.path.join(gt_dir, '*.png'))
srs_imgs_raw = fiFindByWildcard(os.path.join(srs_dir, '*.png'))

gt_imgs = []
srs_imgs = []

print("Start evaluation")

for img_idx in range(n_max):
    gt = os.path.expanduser(os.path.join(gt_dir, f'{901 + img_idx:04d}.png'))

    if gt in gt_imgs_raw:
        gt_imgs.append(gt)
    else:
        raise RuntimeError("Not Found: ", gt)

    if n_samples > 1:
        srs_imgs.append([])
        for i in range(n_samples):
            off = 0
            if os.path.isfile(os.path.join(srs_dir, '000901_sample00000.png')):
                off = 901

            sr = os.path.join(
                srs_dir, f'{off + img_idx:06d}_sample{i:05d}.png')

            if sr in srs_imgs_raw:
                srs_imgs[-1].append(sr)
            else:
                raise RuntimeError("Not Found: ", sr)
    else:
        srs_imgs.append([])
        sr = os.path.join(srs_dir, f'{img_idx:06d}.png')
        if sr in srs_imgs_raw:
            srs_imgs[-1].append(sr)
        else:
            raise RuntimeError("Not Found: ", sr)

print("Found required images.")

loss_fn_alex_sp = lpips.LPIPS(spatial=True)

results = []

for img_idx in range(n_max):
    print(img_idx)
    results.append(lpips_analysis(gt_imgs[img_idx], srs_imgs[img_idx], scale))

df = pd.DataFrame(results)
df_mean = df.mean()

df.to_csv(f"./{name}_full.csv")
df_mean.to_csv(f"./{name}_full_mean.csv")

print()
print(df_mean.to_string())
