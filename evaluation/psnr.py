import torch
import skimage
import numpy as np

def psnr(pred_batch, true_batch):
    pred_batch = pred_batch.detach().cpu().numpy()
    true_batch = true_batch.detach().cpu().numpy()

    psnrs = list()

    for i in range(pred_batch.shape[0]):
        psnrs.append(skimage.metrics.peak_signal_noise_ratio(pred_batch[i], true_batch[i], data_range=1))

    psnr_values = np.stack(psnrs, axis=0)
    return psnr_values
