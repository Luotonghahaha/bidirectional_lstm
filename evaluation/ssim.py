import torch
import torch.nn.functional as F
import numpy as np
import skimage


def ssim(pred_batch, true_batch):
    pred_batch = pred_batch.detach().cpu().numpy()
    true_batch = true_batch.detach().cpu().numpy()

    ssims = list()

    for i in range(pred_batch.shape[0]):
        ssims.append(skimage.metrics.structural_similarity(pred_batch[i], true_batch[i], channel_axis=0, data_range=1))

    ssim_values= np.stack(ssims, axis=0)
    return ssim_values
