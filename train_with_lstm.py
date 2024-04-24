import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config_para import cfg
from data_load.data_loader import subDataset
# from evaluation.psnr import psnr
# from evaluation.ssim import ssim
# from models.models import Encoder, Decoder, RNNModel
from models.lstm_model import Encoder, Decoder, ConvLSTM

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_pro = MakeDataset(data_path)
# make_dataset(path=cfg.data_path, interval=cfg.interval)
# train_data_seq, train_target_inter, test_data_seq, test_target_inter = data_pro.make_data()
# dataset_train = subDataset(train_data_seq, train_target_inter)
# dataset_test = subDataset(test_data_seq, test_target_inter)
dataset_train = subDataset(data_txt='./data/train.txt', data_npy='./data/mnist_train.npy', isTrain=True)
dataset_val = subDataset(data_txt='./data/val.txt', data_npy='./data/mnist_val.npy', isTrain=False)
dataset_test = subDataset(data_txt='./data/test.txt', data_npy='./data/mnist_test.npy', isTrain=False)

# dataset_train = subDataset(data_path='./data', split='train', interval=3)
# dataset_test = subDataset(data_path='./data', split='test', interval=3)
train_dataloader = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, prefetch_factor=2)
val_dataloader = DataLoader(dataset=dataset_val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, drop_last=True, prefetch_factor=2)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, drop_last=True, prefetch_factor=2)
print('All data is ready!')

convlstm_forward = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3,
                            kernel_size=(3, 3),
                            device=device)
convlstm_reverse = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3,
                            kernel_size=(3, 3),
                            device=device)
encoder = Encoder(device)
decoder = Decoder(device)

# model = UnetModel().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.MSELoss().to(device)
params = (list(encoder.parameters()) + list(convlstm_forward.parameters()) + list(convlstm_reverse.parameters()) + list(
    decoder.parameters()))
optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

best_psnr = 0.0
best_ssim = 0.0
best_epoch = 0


def lr_scheduler(opt, epoch, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if (epoch + 1) % lr_decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return opt

# 训练双向RNN: 分别由前后各interval帧生成中间1帧,然后将两个结果加权组合
def train_bidirec(epoch, record, result, train_dataloader, loss_num_per_epoch):
    loss_train = 0.0
    ssim_train = []
    psnr_train = []
    ie_train = 0.0
    pred_train = []
    teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003)

    encoder.train()
    decoder.train()
    convlstm_forward.train()
    convlstm_reverse.train()
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        loss = 0.0
        x_train, y_train = data[0].to(device), data[1].to(device)
        len_x, len_y = x_train.shape[1], y_train.shape[1]
        forward_list = []
        reverse_list = []
        ssim_value = 0.0
        for ii in range(cfg.interval - 1):
            # print('forward_input')
            encoder_output1 = encoder(x_train[:, ii])
            hidden1, output1 = convlstm_forward(encoder_output1, ii == 0)
            decoder_output1 = decoder(output1[-1])
            loss_forward = criterion(decoder_output1, x_train[:, ii + 1])
            # print(loss_forward)
            loss += loss_forward

        # reverse: t+1(in_gt)->t(in_pred), 其中t>interval
        for ij in range(cfg.interval - 1):
            # print('reverse_input')
            encoder_output2 = encoder(x_train[:, len_x - 1 - ij])
            hidden2, output2 = convlstm_forward(encoder_output2, ij == 0)
            decoder_output1 = decoder(output2[-1])
            loss_reverse = criterion(decoder_output1, x_train[:, len_x - 2 - ij])

            # print(loss_reverse)
            loss += loss_reverse

        # forward pred
        lstm_for_input = x_train[:, cfg.interval - 1]  # 正向预测的第一帧由预测的前一帧输入得到
        for ti in range(y_train.shape[1]):
            # print('forward prediction')
            encoder_forward_pred = encoder(lstm_for_input)
            hidden_forward_pred, output_forward_pred = convlstm_forward(encoder_forward_pred, True)
            decoder_forward_pred = decoder(output_forward_pred[-1])
            forward_list.append(decoder_forward_pred)
            loss += criterion(decoder_forward_pred, y_train[:, ti])
            # print(loss_forward_pred)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                lstm_for_input = y_train[:, ti]
            else:
                lstm_for_input = decoder_forward_pred

        # reverse: 对target逐帧从后往前进行预测
        lstm_rev_input = x_train[:, cfg.interval]  # 反向预测的第一帧由预测的后一帧输入得到
        for tj in range(y_train.shape[1]):
            # print('reverse prediction')
            encoder_reverse_pred = encoder(lstm_rev_input)
            hidden_reverse_pred, output_reverse_pred = convlstm_reverse(encoder_reverse_pred, True)
            decoder_reverse_pred = decoder(output_reverse_pred[-1])
            reverse_list.append(decoder_reverse_pred)
            loss += criterion(decoder_reverse_pred, y_train[:, len_y - 1 - tj])

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                lstm_rev_input = y_train[:, len_y - 1 - tj]
            else:
                lstm_rev_input = decoder_reverse_pred
        inter_for = torch.stack(forward_list, dim=1)
        inter_rev = torch.stack(reverse_list, dim=1)
        inter_pred = torch.mean(torch.stack([inter_for, inter_rev], dim=1), dim=1)
        ssim_train.append([ssim(inter_pred[:, t], y_train[:, t]) for t in range(len_y)])
        psnr_train.append([psnr(inter_pred[:, t], y_train[:, t]) for t in range(len_y)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    ssim_train = np.array(ssim_train).mean()
    psnr_train = np.array(psnr_train).mean()
    loss_train = loss_train / (len(train_dataloader) * loss_num_per_epoch)

    record.add_scalar('Loss_Train', loss_train, epoch)
    record.add_scalar('SSIM_Train', ssim_train, epoch)
    record.add_scalar('PSNR_Train', psnr_train, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    lr_scheduler(optimizer, epoch)
    print(
        f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train:.4f}, SSIM: {ssim_train :.4f}, PSNR: {psnr_train :.4f}')
    result.write(
        f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train:.4f}, SSIM: {ssim_train :.4f}, PSNR: {psnr_train :.4f}')


def test_bidirec(epoch, record, result, train_dataloader, loss_num_per_epoch):
    global best_psnr, best_ssim, best_epoch
    loss = 0.0
    ssim_test = []
    psnr_test = []
    ie_train = 0.0
    pred_train = []

    encoder.eval()
    convlstm_forward.eval()
    convlstm_reverse.eval()
    decoder.eval()

    for i, data in tqdm(enumerate(train_dataloader), total=len(test_dataloader)):
        x_test, y_test = data[0].to(device), data[1].to(device)
        len_x, len_y = x_test.shape[1], y_test.shape[1]
        forward_list = []
        reverse_list = []
        ssim_value = 0.0
        for ii in range(cfg.interval - 1):
            # print('forward_input')
            encoder_output1 = encoder(x_test[:, ii])
            hidden1, output1 = convlstm_forward(encoder_output1, ii == 0)
            decoder_output1 = decoder(output1[-1])
            loss_forward = criterion(decoder_output1, x_test[:, ii + 1])
            # print(loss_forward)
            loss += loss_forward

        # reverse: t+1(in_gt)->t(in_pred), 其中t>interval
        for ij in range(cfg.interval - 1):
            # print('reverse_input')
            encoder_output2 = encoder(x_test[:, len_x - 1 - ij])
            hidden2, output2 = convlstm_forward(encoder_output2, ij == 0)
            decoder_output1 = decoder(output2[-1])
            loss_reverse = criterion(decoder_output1, x_test[:, len_x - 2 - ij])

            # print(loss_reverse)
            loss += loss_reverse

        # forward pred
        lstm_for_input = x_test[:, cfg.interval - 1]  # 正向预测的第一帧由预测的前一帧输入得到
        for ti in range(y_test.shape[1]):
            # print('forward prediction')
            encoder_forward_pred = encoder(lstm_for_input)
            hidden_forward_pred, output_forward_pred = convlstm_forward(encoder_forward_pred, True)
            decoder_forward_pred = decoder(output_forward_pred[-1])
            forward_list.append(decoder_forward_pred)
            loss += criterion(decoder_forward_pred, y_test[:, ti])
            lstm_for_input = decoder_forward_pred
            # print(loss_forward_pred)

        # reverse: 对target逐帧从后往前进行预测
        lstm_rev_input = x_test[:, cfg.interval]  # 反向预测的第一帧由预测的后一帧输入得到
        for tj in range(y_test.shape[1]):
            # print('reverse prediction')
            encoder_reverse_pred = encoder(lstm_rev_input)
            hidden_reverse_pred, output_reverse_pred = convlstm_reverse(encoder_reverse_pred, True)
            decoder_reverse_pred = decoder(output_reverse_pred[-1])
            reverse_list.append(decoder_reverse_pred)
            loss += criterion(decoder_reverse_pred, y_test[:, len_y - 1 - tj])
            lstm_rev_input = decoder_reverse_pred

        inter_for = torch.stack(forward_list, dim=1)
        inter_rev = torch.stack(reverse_list, dim=1)
        inter_pred = torch.mean(torch.stack([inter_for, inter_rev], dim=1), dim=1)
        ssim_test.append([ssim(inter_pred[:, t], y_test[:, t]) for t in range(len_y)])
        psnr_test.append([psnr(inter_pred[:, t], y_test[:, t]) for t in range(len_y)])

    ssim_test = np.array(ssim_test).mean()
    psnr_test = np.array(psnr_test).mean()
    loss_test = loss / (len(test_dataloader) * loss_num_per_epoch)

    record.add_scalar('Loss_Test', loss_test, epoch)
    record.add_scalar('SSIM_Test', ssim_test, epoch)
    record.add_scalar('PSNR_Test', psnr_test, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    lr_scheduler(optimizer, epoch)
    print(
        f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test :.4f}, PSNR: {psnr_test :.4f}')
    result.write(
        f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test :.4f}, PSNR: {psnr_test :.4f}')
    if psnr_test > best_psnr:
        best_psnr = psnr_test
        best_epoch = epoch
        # path = os.path.join(save_path, 'ckpt',
        #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
        print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test:.4f}')
        result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test:.4f}\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'best_psnr_encoder.pth')
        path_lstm_forward = os.path.join(save_path, 'ckpt', f'last_lstm_forward.pth')
        path_lstm_reverse = os.path.join(save_path, 'ckpt', f'last_lstm_reverse.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'best_psnr_decoder.pth')

        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': convlstm_forward.state_dict()}, path_lstm_forward)
        torch.save({'state_dict': convlstm_reverse.state_dict()}, path_lstm_reverse)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
    else:
        print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}')
        result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}\n')

    if ssim_test > best_ssim:
        best_ssim = ssim_test
        best_epoch = epoch
        # path = os.path.join(save_path, 'ckpt',
        #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
        print(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_test:.4f}, best_ssim:{best_ssim:.4f}')
        result.write(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_test:.4f}, best_ssim:{best_ssim:.4f}\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'best_ssim_encoder.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'best_ssim_decoder.pth')
        path_lstm_forward = os.path.join(save_path, 'ckpt', f'last_lstm_forward.pth')
        path_lstm_reverse = os.path.join(save_path, 'ckpt', f'last_lstm_reverse.pth')

        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': convlstm_forward.state_dict()}, path_lstm_forward)
        torch.save({'state_dict': convlstm_reverse.state_dict()}, path_lstm_reverse)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
    else:
        print(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}')
        result.write(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}\n')


if __name__ == '__main__':
    name = 'BidirecRNN'
    loss_num_per_epoch = (cfg.interval - 1) * 2 + cfg.target_num * 2
    # 'interval': 2,
    # 'target_num': 1,
    # name = 'BidirecRNN'
    seed = int(time.time())
    save_path = f'./Logs/{name}_{seed}'  # 保存模型
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'ckpt'))
    record = SummaryWriter(log_dir=save_path)
    # result = open(result_save_path, 'w')
    record_file = save_path + '/output.txt'
    with open(record_file, 'a') as result:
        for e in range(cfg.epochs):
            result = open(record_file, 'a')
            train_bidirec(e + 1, record, result, train_dataloader, loss_num_per_epoch)
            # loss, ssim_eval, psnr_eval, ie_eval = test(e, record, val_dataloader)
            test_bidirec(e + 1, record, result, val_dataloader, loss_num_per_epoch)
        # 保存最后一次的模型
        print('Test on test dataset:')
        result.write('Test on test dataset:\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'last_encoder.pth')
        path_lstm_forward = os.path.join(save_path, 'ckpt', f'last_lstm_forward.pth')
        path_lstm_reverse = os.path.join(save_path, 'ckpt', f'last_lstm_reverse.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'last_decoder.pth')
        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': convlstm_forward.state_dict()}, path_lstm_forward)
        torch.save({'state_dict': convlstm_reverse.state_dict()}, path_lstm_reverse)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
        test_bidirec(cfg.epochs, record, result, test_dataloader, loss_num_per_epoch)
        print('Accomplished!')
        result.write('Accomplished!')