import torch
import torch.nn as nn
# from models.pre_model import UnetModel
from models.models import Encoder, Decoder, RNNModel
from evaluation.ssim import ssim
from evaluation.psnr import psnr
from config_para import cfg
import numpy as np
from torchvision import transforms as T
import os

index = 0
sample_path = cfg.test_path
test_npy_path = cfg.test_npy_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform = T.Compose([
                            T.ToPILImage(),
                            T.ToTensor(),
                            T.Normalize((0.1307,), (0.3081,))
                            ])


def demo_sample(index, sample_path, test_npy_path):
    test_npy = np.load(test_npy_path)
    id_file = open(sample_path, 'r')
    video = []
    for line in id_file:
        line_ = [int(i) for i in line.strip().split(',')]
        if line_[0] == index:
            video_i = test_npy[index, line_[1:]]
            video.append(video_i)
    video_np = np.array(video)
    video_tensor = torch.zeros(video_np.shape)
    for i in range(video_np.shape[0]):
        for j in range(video_np.shape[1]):
            video_tensor[i, j] = test_transform(video_np[i, j])
    data = torch.tensor(video_np[:, 0:-1, :, :])
    target = torch.tensor(video_np[:, -1, :, :]).unsqueeze(1)
    return data, target


demo_data, demo_target = demo_sample(index, sample_path, test_npy_path)
model_para = './Logs/UnetModel_1710248296/ckpt/best_psnr.pth'

# model = UnetModel().to(device)
encoder = Encoder().to(device)
latent_model = RNNModel(cfg.hidden_rnn, cfg.hidden_rnn).to(device)
decoder = Decoder().to(device)
h_0 = torch.zeros(1, 6, cfg.hidden_rnn).to(device)

# encoder.load_state_dict(torch.load(model_para))
# latent_model.load_state_dict(torch.load(model_para))
# decoder.load_state_dict(torch.load(model_para))

# pred = model(demo_data.float())


# # 单向RNN
# # print(demo_data.shape)
# hidden_x = encoder(demo_data.float().unsqueeze(2).transpose(0, 1))
# # print(hidden_x.shape)
# hidden_y, _ = latent_model(hidden_x.view(cfg.interval * 2, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
# pred_unirnn = decoder(hidden_y)

# 双向RNN
hidden_x = encoder(demo_data.float().unsqueeze(2).transpose(0, 1))
forward_y, _ = latent_model(hidden_x[0: 2].view(cfg.interval, 6, cfg.hidden_rnn * 1 * 1), h_0)
reverse_y, _ = latent_model(torch.flip(hidden_x[2:], dims=[0]).view(cfg.interval, 6, cfg.hidden_rnn * 1 * 1), h_0)
hidden_y = cfg.epsilon * forward_y + (1 - cfg.epsilon) * reverse_y

pred_birnn = decoder(hidden_y.view(6, cfg.hidden_rnn, 1, 1))

# # 单向RNN
# uni_rnn_demo_path = './out/generate/uni_rnn'
# toPIL = T.ToPILImage()
# if not os.path.exists(uni_rnn_demo_path):
#     os.makedirs(uni_rnn_demo_path)
# for i in range(demo_data.shape[0]):
#     for j in range(cfg.interval):
#         toPIL(demo_data[i, j]).save(uni_rnn_demo_path + '/' + str((cfg.interval + 1) * i + j) + '.png')
#     toPIL(pred_unirnn[i]).save(uni_rnn_demo_path + '/' + str((cfg.interval + 1) * i + 1) + '_' + str((cfg.interval + 1) * i + 3) + '.png')
# for i in range(cfg.interval):
#      toPIL(demo_data[-1, cfg.interval+i]).save(uni_rnn_demo_path + '/' + str(cfg.T - (i+1)) + '.png')

# 双向RNN
bi_rnn_demo_path = './out/generate/bi_rnn'
toPIL = T.ToPILImage()
if not os.path.exists(bi_rnn_demo_path):
    os.makedirs(bi_rnn_demo_path)

for i in range(demo_data.shape[0]):
    for j in range(cfg.interval):
        toPIL(demo_data[i, j]).save(bi_rnn_demo_path + '/' + str((cfg.interval + 1) * i + j) + '.png')
    toPIL(pred_birnn[i]).save(bi_rnn_demo_path + '/' + str((cfg.interval + 1) * i + 1) + '_' + str((cfg.interval + 1) * i + 3) + '.png')
for i in range(cfg.interval):
     toPIL(demo_data[-1, cfg.interval+i]).save(bi_rnn_demo_path + '/' + str(cfg.T - (i+1)) + '.png')
