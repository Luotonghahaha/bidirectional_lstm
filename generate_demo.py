import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms as T

import os
from config_para import cfg
# from models.pre_model import UnetModel
from models.lstm_model import Encoder, Decoder, ConvLSTM

index = 0
sample_path = cfg.test_path
test_npy_path = cfg.test_npy_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
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
    data = torch.tensor(video_np[:, 0:-1, :, :]).unsqueeze(2)
    target = torch.tensor(video_np[:, -1, :, :]).unsqueeze(1)
    return data, target


demo_data, demo_target = demo_sample(index, sample_path, test_npy_path)

# 实例化模型
convlstm_forward = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3,
                            kernel_size=(3, 3),
                            device=device)
encoder = Encoder(device)
decoder = Decoder(device)

# 加载已经保存的模型
checkpoint_encoder = torch.load('./Logs/ckpt/last_encoder.pth', map_location=device)
checkpoint_lstm_forward = torch.load('./Logs/ckpt/last_lstm_forward.pth', map_location=device)
checkpoint_decoder = torch.load('./Logs/ckpt/last_decoder.pth', map_location=device)

encoder.load_state_dict(checkpoint_encoder['state_dict'])
convlstm_forward.load_state_dict(checkpoint_lstm_forward['state_dict'])
decoder.load_state_dict(checkpoint_decoder['state_dict'])

pred_list = []
lstm_for_input = demo_data[:, cfg.interval - 1]
for ti in range(cfg.target_num):
    # print('forward prediction')
    encoder_forward_pred = encoder(lstm_for_input.to(torch.float32))
    hidden_forward_pred, output_forward_pred = convlstm_forward(encoder_forward_pred, ti == 0)
    decoder_forward_pred = decoder(output_forward_pred[-1])
    pred_list.append(decoder_forward_pred)

    # print(loss_forward_pred)
    lstm_for_input = decoder_forward_pred

inter_pred = torch.stack(pred_list, dim=1)

fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 6))
# 显示 ground truth 图像
for i, ax in enumerate(axes[0]):
    ax.imshow(demo_target[i, 0].squeeze(0), cmap='gray')
    ax.set_title('T='+str(i+cfg.interval))
    ax.axis('off')

# 显示预测图像
for i, ax in enumerate(axes[1]):
    ax.imshow(inter_pred[i, 0].squeeze(0).detach().numpy(), cmap='gray')
    ax.set_title('T='+str(i+cfg.interval))
    ax.axis('off')

plt.suptitle("Ground Truth vs. Prediction")
plt.tight_layout()

save_folder = './out'
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, 'ground_truth_vs_prediction.png')
plt.savefig(save_path)

