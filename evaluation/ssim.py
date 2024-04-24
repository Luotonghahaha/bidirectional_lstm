import numpy as np
import skimage


def ssim(pred_batch, true_batch):
    pred_batch = pred_batch.detach().cpu().numpy()
    true_batch = true_batch.detach().cpu().numpy()

    ssims = list()

    for t in range(pred_batch.shape[1]):
        ssims.append(
            [skimage.metrics.structural_similarity(pred_batch[i, t], true_batch[i, t], channel_axis=0, data_range=1) for i in range(pred_batch.shape[0])])

    ssim_values = np.stack(ssims, axis=1)
    return ssim_values
