import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms as T

import os
from config_para import cfg
# from models.pre_model import UnetModel
from models.lstm_model import Encoder, Decoder, ConvLSTM
from models.simvp_model import SimVP_Model
from models.simvp_model import Encoder, Decoder, ODENet
import random

index = 0
seq_len_total = 11
seq_len = 7
total_len = 20
start_index = (cfg.interval - 1) * (cfg.target_num + 1) + 1
end_index = start_index + cfg.target_num
test_npy_path = './datasets/moving_mnist/mnist_test_seq.npy'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])


def demo_sample(index, seq_len_total, seq_len, T, test_npy_path):
    test_npy = np.load(test_npy_path)
    video = []
    for i in range(T-seq_len_total+1):
        video_i = test_npy[i:i+seq_len_total, index]
        video.append(video_i)
    video_np = np.array(video)
    images_interval = video_np[:, ::cfg.target_num + 1]
    images_target = video_np[:, start_index:end_index]
    images = np.concatenate([images_interval, images_target], axis=1)
    images = torch.from_numpy(images / 255.0).contiguous().float().unsqueeze(2)
    data = images[:, :cfg.interval * 2]
    target = images[:, -cfg.target_num:, ...]
    return data, target


# def demo_sample(index, sample_path, test_npy_path):
#     test_npy = np.load(test_npy_path)
#     id_file = open(sample_path, 'r')
#     video = []
#     for line in id_file:
#         line_ = [int(i) for i in line.strip().split(',')]
#         if line_[0] == index:
#             video_i = test_npy[index, line_[1:]]
#
#             video.append(video_i)
#     video_np = np.array(video)
#     video_tensor = torch.zeros(video_np.shape)
#     for i in range(video_np.shape[0]):
#         for j in range(video_np.shape[1]):
#             video_tensor[i, j] = test_transform(video_np[i, j])
#     data = torch.tensor(video_np[:, 0:-1, :, :]).unsqueeze(2)
#     target = torch.tensor(video_np[:, -1, :, :]).unsqueeze(1)
#     return data, target

demo_data, demo_target = demo_sample(index, seq_len_total, seq_len, total_len, test_npy_path)

in_shape = [cfg.interval, cfg.channel, cfg.height, cfg.weight]
simvp_model = SimVP_Model(in_shape)
checkpoint_model = torch.load('./Logs/ckpt/best_ssim_simvp.pth', map_location=device)
simvp_model.load_state_dict(checkpoint_model['state_dict'])

enc = Encoder(1, 16, 4, 3, act_inplace=False)
dec = Decoder(16, 1, 4, 3, act_inplace=False)
checkpoint_enc = torch.load('./Logs/ckpt/best_ssim_enc.pth', map_location=device)
checkpoint_dec = torch.load('./Logs/ckpt/best_ssim_dec.pth', map_location=device)
enc.load_state_dict(checkpoint_enc['state_dict'])
dec.load_state_dict(checkpoint_dec['state_dict'])


# pred = simvp_model(demo_data[:, :cfg.interval].to(torch.float))[:, :cfg.target_num]

x_test, y_test = demo_data[:, :cfg.interval], demo_target  # [B, T, C, H, W]
B, T, C, H, W = x_test.shape
x = x_test.contiguous().view(B * T, C, H, W)

embed, skip = enc(x)
pred = dec(embed).view(B, T, C, H, W)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))
# 显示 ground truth 图像
for i, ax in enumerate(axes[0]):
    ax.imshow(demo_data[0, i].squeeze(0), cmap='gray')
    ax.set_title('T='+str(i+cfg.interval))
    ax.axis('off')

# 显示预测图像
for i, ax in enumerate(axes[1]):
    ax.imshow(pred[0, i].squeeze(0).detach().numpy(), cmap='gray')
    ax.set_title('T='+str(i+cfg.interval))
    ax.axis('off')

plt.suptitle("Ground Truth vs. Prediction")
plt.tight_layout()

save_folder = './out'
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, 'ground_truth_vs_predictionq_ed.png')
plt.savefig(save_path)

