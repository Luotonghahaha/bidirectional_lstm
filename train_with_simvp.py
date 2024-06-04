import os
import time

import numpy as np
import torch
import torch.nn as nn  # from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config_para import cfg
from data_load.data_loader import subDataset
from evaluation.psnr import psnr
from evaluation.ssim import ssim
# from models.models import Encoder, Decoder, RNNModel
from models.simvp_model import SimVP_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_pro = MakeDataset(data_path)
# make_dataset(path=cfg.data_path, interval=cfg.interval)
# train_data_seq, train_target_inter, test_data_seq, test_target_inter = data_pro.make_data()
# dataset_train = subDataset(train_data_seq, train_target_inter)
# dataset_test = subDataset(test_data_seq, test_target_inter)
dataset_train = subDataset(root_path='./data_load', interval=cfg.interval, target_num=cfg.target_num,
                           channel=cfg.channel, image_size=cfg.image_size,
                           isTrain=True, use_augment=True)
dataset_test = subDataset(root_path='./data_load', interval=cfg.interval, target_num=cfg.target_num,
                          channel=cfg.channel, image_size=cfg.image_size,
                          isTrain=False, use_augment=False)

# dataset_train = subDataset(data_path='./data', split='train', interval=3)
# dataset_test = subDataset(data_path='./data', split='test', interval=3)
train_dataloader = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, prefetch_factor=2)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, drop_last=True, prefetch_factor=2)
print('All data is ready!')


# def lr_scheduler(opt, epoch, lr_decay_epoch=50):
#     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
#     if (epoch + 1) % lr_decay_epoch == 0:
#         if opt == 'optimizer_uni':
#             for param_group in opt.param_groups:
#                 param_group['lr'] = param_group['lr'] * 0.1
#         elif opt == 'optimizer_bi':
#             for param_group in opt.param_groups:
#                 param_group['lr'] = param_group['lr'] * 0.1
#     return opt


# 训练单向RNN: 分别由前后各interval帧生成中间1帧,然后将两个结果加权组合
def train_unidirec(epoch, record, result, train_dataloader):
    loss_train = []
    ssim_train = []
    psnr_train = []
    ie_train = 0.0

    simvp_model.train()
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x_train, y_train = data[0].to(device), data[1].to(device)  # [B, T, C, H, W]
        # x_T, y_T = x_train.shape[1], y_train.shape[1]
        pred_list = []
        pred = simvp_model(x_train[:, :cfg.interval])[:, :cfg.target_num]
        loss = criterion(pred, y_train)

        optimizer_uni.zero_grad()
        loss.backward()
        optimizer_uni.step()

        loss_train.append(loss.item())
        ssim_train.append(ssim(pred, y_train))
        psnr_train.append(psnr(pred, y_train))

    ssim_train_mean = np.stack(ssim_train, axis=0).mean()
    psnr_train_mean = np.stack(psnr_train, axis=0).mean()
    loss_train_mean = np.stack(loss_train, axis=0).mean()

    record.add_scalar('Loss_Train', loss_train_mean, epoch)
    record.add_scalar('SSIM_Train', ssim_train_mean, epoch)
    record.add_scalar('PSNR_Train', psnr_train_mean, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    print(
        f'Train on Epoch {epoch}/{cfg.epochs}, Loss: {loss_train_mean:.4f}, SSIM: {ssim_train_mean :.4f}, PSNR: {psnr_train_mean :.4f}')
    result.write(
        f'Test on Epoch {epoch}/{cfg.epochs}, Loss: {loss_train_mean:.4f}, SSIM: {ssim_train_mean :.4f}, PSNR: {psnr_train_mean :.4f}')


def test_unidirec(epoch, record, result, test_dataloader):
    global best_psnr, best_ssim, best_epoch
    loss_test = []
    ssim_test = []
    psnr_test = []
    ie_train = 0.0
    pred_train = []

    simvp_model.eval()
    with torch.no_grad():

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x_test, y_test = data[0].to(device), data[1].to(device)  # [B, T, C, H, W]
            # x_T, y_T = x_test.shape[1], y_test.shape[1]
            pred = simvp_model(x_test[:, :cfg.interval])[:, :cfg.target_num]
            loss = criterion(pred, y_test)

            loss_test.append(loss.item())
            ssim_test.append(ssim(pred, y_test))
            psnr_test.append(psnr(pred, y_test))


        ssim_test_mean = np.stack(ssim_test, axis=0).mean()
        psnr_test_mean = np.stack(psnr_test, axis=0).mean()
        loss_test = np.stack(loss_test, axis=0).mean()
        scheduler_uni.step(loss_test)

        record.add_scalar('Loss_Test', loss_test, epoch)
        record.add_scalar('SSIM_Test', ssim_test_mean, epoch)
        record.add_scalar('PSNR_Test', psnr_test_mean, epoch)
        # record.add_scalar('IE_Train', ie_train, epoch)
        # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
        #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
        #         IE: {psnr_sum / (i + 1):.2f}')

        print(
            f'Test on epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test_mean :.4f}, PSNR: {psnr_test_mean :.4f}')
        result.write(
            f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test_mean :.4f}, PSNR: {psnr_test_mean :.4f}')
        if psnr_test_mean > best_psnr:
            best_psnr = psnr_test_mean
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test_mean:.4f}')
            result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test_mean:.4f}\n')

            path_save = os.path.join(save_path, 'ckpt', f'best_psnr_simvp.pth')
            torch.save({'state_dict': simvp_model.state_dict()}, path_save)
        else:
            print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}')
            result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}\n')

        if ssim_test_mean > best_ssim:
            best_ssim = ssim_test_mean
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_test_mean:.4f}, best_ssim:{best_ssim:.4f}')
            result.write(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_test_mean:.4f}, best_ssim:{best_ssim:.4f}\n')

            path_save = os.path.join(save_path, 'ckpt', f'best_ssim_simvp.pth')
            torch.save({'state_dict': simvp_model.state_dict()}, path_save)
        else:
            print(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}')
            result.write(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}\n')


# 训练双向RNN: 分别由前后各interval帧生成中间1帧,然后将两个结果加权组合
def train_bidirec(epoch, record, result, train_dataloader):
    loss_train = []
    ssim_train = []
    psnr_train = []
    ie_train = 0.0
    pred_train = []
    teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003)

    simvp_model_forward.train()
    simvp_model_reverse.train()
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        loss = 0.0
        x_train, y_train = data[0].to(device), data[1].to(device)
        forward_list = []
        reverse_list = []
        ssim_value = 0.0

        pred_forward = simvp_model_forward(x_train[:, :cfg.interval])[:, :cfg.target_num]
        pred_reverse = simvp_model_reverse(torch.flip(x_train[:, cfg.interval:], dims=[1]))[:, :cfg.target_num]
        pred = cfg.epsilon * pred_forward + (1 - cfg.epsilon) * pred_reverse
        loss = criterion(pred, y_train)

        optimizer_bi.zero_grad()
        loss.backward()
        optimizer_bi.step()

        loss_train.append(loss.item())
        ssim_train.append(ssim(pred, y_train))
        psnr_train.append(psnr(pred, y_train))

    ssim_mean_epoch = np.stack(ssim_train, axis=0).mean()
    psnr_mean_epoch = np.stack(psnr_train, axis=0).mean()
    loss_train = np.stack(loss_train, axis=0).mean()

    record.add_scalar('Loss_Train', loss_train, epoch)
    record.add_scalar('SSIM_Train', ssim_mean_epoch, epoch)
    record.add_scalar('PSNR_Train', psnr_mean_epoch, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    print(
        f'Train on Epoch {epoch}/{cfg.epochs}, Loss: {loss_train:.4f}, SSIM: {ssim_mean_epoch :.4f}, PSNR: {psnr_mean_epoch :.4f}')
    result.write(
        f'Train on Epoch {epoch}/{cfg.epochs}, Loss: {loss_train:.4f}, SSIM: {ssim_mean_epoch :.4f}, PSNR: {psnr_mean_epoch :.4f}')


def test_bidirec(epoch, record, result, test_dataloader):
    global best_psnr, best_ssim, best_epoch
    loss_test = []
    ssim_test = []
    psnr_test = []
    ie_train = 0.0
    pred_train = []

    simvp_model_forward.eval()
    simvp_model_reverse.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x_test, y_test = data[0].to(device), data[1].to(device)
            forward_list = []
            reverse_list = []
            ssim_value = 0.0
            pred_forward = simvp_model_forward(x_test[:, :cfg.interval])[:, :cfg.target_num]
            pred_reverse = simvp_model_reverse(torch.flip(x_test[:, cfg.interval:], dims=[1]))[:, :cfg.target_num]
            pred = cfg.epsilon * pred_forward + (1 - cfg.epsilon) * pred_reverse
            loss = criterion(pred, y_test)

            loss_test.append(loss.item())
            ssim_test.append(ssim(pred, y_test))
            psnr_test.append(psnr(pred, y_test))

        ssim_test_mean = np.stack(ssim_test, axis=0).mean()
        psnr_test_mean = np.stack(psnr_test, axis=0).mean()
        loss_test = np.stack(loss_test, axis=0).mean()
        scheduler_bi.step(loss_test)

        record.add_scalar('Loss_Test', loss_test, epoch)
        record.add_scalar('SSIM_Test', ssim_test_mean, epoch)
        record.add_scalar('PSNR_Test', psnr_test_mean, epoch)
        # record.add_scalar('IE_Train', ie_train, epoch)
        # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
        #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
        #         IE: {psnr_sum / (i + 1):.2f}')
        print(
            f'Test on Epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test_mean :.4f}, PSNR: {psnr_test_mean :.4f}')
        result.write(
            f'Test on Epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test_mean :.4f}, PSNR: {psnr_test_mean :.4f}')
        if psnr_test_mean > best_psnr:
            best_psnr = psnr_test_mean
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test_mean:.4f}')
            result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test_mean:.4f}\n')

            path_save_forward = os.path.join(save_path, 'ckpt', f'best_simvp_psnr_forward.pth')
            path_save_reverse = os.path.join(save_path, 'ckpt', f'best_simvp_psnr_reverse.pth')

            torch.save({'state_dict': simvp_model_forward.state_dict()}, path_save_forward)
            torch.save({'state_dict': simvp_model_reverse.state_dict()}, path_save_reverse)
        else:
            print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}')
            result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}\n')

        if ssim_test_mean > best_ssim:
            best_ssim = ssim_test_mean
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_test_mean:.4f}, best_ssim:{best_ssim:.4f}')
            result.write(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_test_mean:.4f}, best_ssim:{best_ssim:.4f}\n')

            path_save_forward = os.path.join(save_path, 'ckpt', f'best_simvp_ssim_forward.pth')
            path_save_reverse = os.path.join(save_path, 'ckpt', f'best_simvp_ssim_reverse.pth')

            torch.save({'state_dict': simvp_model_forward.state_dict()}, path_save_forward)
            torch.save({'state_dict': simvp_model_reverse.state_dict()}, path_save_reverse)
        else:
            print(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}')
            result.write(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}\n')


if __name__ == '__main__':
    print(cfg.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss().to(device)
    # 'interval': 2, 'target_num': 1, cfg.name = 'BidirecRNN'
    in_shape = [cfg.interval, cfg.channel, cfg.height, cfg.weight]
    seed = int(time.time())
    save_path = f'./Logs/{cfg.name}_{seed}'  # 保存模型
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'ckpt'))
    record = SummaryWriter(log_dir=save_path)
    # result = open(result_save_path, 'w')
    record_file = save_path + '/output.txt'
    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch = 0
    if cfg.name == 'UnidirecLSTM':  # unidirection
        simvp_model = SimVP_Model(in_shape).to(device)
        optimizer_uni = torch.optim.Adam(simvp_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler_uni = OneCycleLR(optimizer_uni, mode='min', patience=2, factor=0.1, verbose=True)
        print('training with UnidirectLSTM!')
        with open(record_file, 'a') as result:
            for e in range(cfg.epochs):
                result = open(record_file, 'a')
                train_unidirec(e + 1, record, result, train_dataloader)
                # 在val数据上进行测试
                test_unidirec(e + 1, record, result, test_dataloader)

            # 保存最后一次模型
            path_save = os.path.join(save_path, 'ckpt', f'last_simvp.pth')
            torch.save({'state_dict': simvp_model.state_dict()}, path_save)
            print('Test on test dataset:')
            result.write('Test on test dataset:\n')
            # 在test数据中进行测试
            test_unidirec(cfg.epochs, record, result, test_dataloader)
            print('Accomplished!')
            result.write('Accomplished!')
    elif cfg.name == 'BidirecLSTM':
        # bidirection
        simvp_model_forward = SimVP_Model(in_shape).to(device)
        simvp_model_reverse = SimVP_Model(in_shape).to(device)
        param_bi = (list(simvp_model_forward.parameters()) + list(simvp_model_reverse.parameters()))
        optimizer_bi = torch.optim.Adam(param_bi, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler_bi = ReduceLROnPlateau(optimizer_bi, mode='min', patience=2, factor=0.1, verbose=True)
        # 定义全局最优pnsr和ssim，以及对应的epoch
        print('training on BidirecLSTM')
        with open(record_file, 'a') as result:
            for e in range(cfg.epochs):
                result = open(record_file, 'a')

                train_bidirec(e + 1, record, result, train_dataloader)
                # 在val数据上进行验证
                test_bidirec(e + 1, record, result, test_dataloader)

            # 保存最后一次的模型
            path_save_forward = os.path.join(save_path, 'ckpt', f'last_simvp_forward.pth')
            path_save_reverse = os.path.join(save_path, 'ckpt', f'last_simvp_reverse.pth')

            torch.save({'state_dict': simvp_model_forward.state_dict()}, path_save_forward)
            torch.save({'state_dict': simvp_model_reverse.state_dict()}, path_save_reverse)
            # 在test数据上进行测试
            print('Test on test dataset:')
            result.write('Test on test dataset:\n')
            test_bidirec(cfg.epochs, record, result, test_dataloader)
            print('Accomplished!')
            result.write('Accomplished!')
