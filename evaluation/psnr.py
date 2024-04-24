import torch
import skimage
import numpy as np

def psnr(pred_batch, true_batch):
    pred_batch = pred_batch.detach().cpu().numpy()
    true_batch = true_batch.detach().cpu().numpy()

    psnrs = list()

    for t in range(pred_batch.shape[1]):
        psnrs.append([skimage.metrics.peak_signal_noise_ratio(pred_batch[i, t], true_batch[i, t], data_range=1) for i in range(pred_batch.shape[0])])

    psnr_values = np.stack(psnrs, axis=1)
    return psnr_values
