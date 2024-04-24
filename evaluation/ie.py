import torch
from skimage.exposure import histogram

def information_entropy(frame):
    """计算单个帧的信息熵"""
    p = histogram(frame)
    return -torch.sum(p * torch.log(p))


def ie(frames):
    """
    frames: shape=(num_frames, h, w, c)
    """
    print(frames.shape)
    exit()
    ie_scores = []

    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        # 计算两个帧的信息熵
        ent1 = information_entropy(frame1)
        ent2 = information_entropy(frame2)

        # 计算信息熵差异率
        ie = abs(ent1 - ent2) / max(ent1, ent2)

        ie_scores.append(ie)

    return torch.mean(ie_scores)