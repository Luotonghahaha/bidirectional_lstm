import os
import time
import sys
sys.path.append("..")
sys.path.append(".")

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
from models.simvp_model import Encoder, Decoder, ODENet

# data_pro = MakeDataset(data_path)
# make_dataset(path=cfg.data_path, interval=cfg.interval)
# train_data_seq, train_target_inter, test_data_seq, test_target_inter = data_pro.make_data()
# dataset_train = subDataset(train_data_seq, train_target_inter)
# dataset_test = subDataset(test_data_seq, test_target_inter)
dataset_train = subDataset(root_path=cfg.root_path, interval=cfg.interval, target_num=cfg.target_num,
                           channel=cfg.channel, image_size=cfg.image_size,
                           isTrain=True, use_augment=True)
dataset_test = subDataset(root_path=cfg.root_path, interval=cfg.interval, target_num=cfg.target_num,
                          channel=cfg.channel, image_size=cfg.image_size,
                          isTrain=False, use_augment=False)

# dataset_train = subDataset(data_path='./data', split='train', interval=3)
# dataset_test = subDataset(data_path='./data', split='test', interval=3)
train_dataloader = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=1, drop_last=True, prefetch_factor=2)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=1, drop_last=True, prefetch_factor=2)
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

    enc.train()
    dec.train()
    latent.train()
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x_train, y_train = data[0][:, :cfg.interval].to(device), data[1].to(device)  # [B, T, C, H, W]
        B, T, C, H, W = x_train.shape
        x = x_train.contiguous().view(B * T, C, H, W)
        embed, skip = enc(x)

        _, C1_, H1_, W1_ = embed.shape
        z_ode = embed.view(B, T, C1_, H1_, W1_)


        # _, C2_, H2_, W2_ = skip.shape
        # h = skip.view(B, T, C2_, H2_, W2_)
        # h = torch.mean(h, dim=1)

        diff_z = torch.mean(torch.diff(z_ode, dim=1), dim=1)
        hid = latent(diff_z)

        pred = dec(hid).unsqueeze(1)
        loss = criterion(pred, y_train)

        # # 不加ode
        # embed, skip = enc(x)
        # pred = dec(embed).view(B, T, C, H, W)
        # loss = criterion(pred, x_train)

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
        f'Train on Epoch {epoch}/{cfg.epochs}, Loss: {loss_train_mean:.4f}, SSIM: {ssim_train_mean :.4f}, PSNR: {psnr_train_mean :.4f}')

def test_unidirec(epoch, record, result, test_dataloader):
    global best_psnr, best_ssim, best_epoch
    loss_test = []
    ssim_test = []
    psnr_test = []
    ie_train = 0.0
    pred_train = []

    enc.eval()
    latent.eval()
    dec.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x_test, y_test = data[0][:, :cfg.interval].to(device), data[1].to(device)  # [B, T, C, H, W]
            B, T, C, H, W = x_test.shape
            x = x_test.contiguous().view(B * T, C, H, W)

            # 加ode
            embed, skip = enc(x)

            _, C1_, H1_, W1_ = embed.shape
            z_ode = embed.view(B, T, C1_, H1_, W1_)
            # z_dec = embed.reshape(B * T, C1_, H1_, W1_)
            # h_dec = dec(z_dec).view(B, T, C, H, W)

            # _, C2_, H2_, W2_ = skip.shape
            # h = skip.view(B, T, C2_, H2_, W2_)
            # h = torch.mean(h, dim=1)

            diff_z = torch.mean(torch.diff(z_ode, dim=1), dim=1)
            hid = latent(diff_z)
            pred = dec(hid).unsqueeze(1)
            loss = criterion(pred, y_test)

            # # 不加ode
            # embed, skip = enc(x)
            # pred = dec(embed).view(B, T, C, H, W)
            # loss = criterion(pred, x_test)

            loss_test.append(loss.item())
            ssim_test.append(ssim(pred, y_test))
            psnr_test.append(psnr(pred, y_test))

        ssim_test_mean = np.stack(ssim_test, axis=0).mean()
        psnr_test_mean = np.stack(psnr_test, axis=0).mean()
        loss_test = np.stack(loss_test, axis=0).mean()
        scheduler_uni.step()

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
            f'Test on epoch {epoch}/{cfg.epochs}, Loss: {loss_test:.4f}, SSIM: {ssim_test_mean :.4f}, PSNR: {psnr_test_mean :.4f}')
        if psnr_test_mean > best_psnr:
            best_psnr = psnr_test_mean
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test_mean:.4f}')
            result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_test_mean:.4f}\n')

            path_save_enc = os.path.join(save_path, 'ckpt', f'best_psnr_enc.pth')
            # path_save_latent = os.path.join(save_path, 'ckpt', f'best_psnr_latent.pth')
            path_save_dec = os.path.join(save_path, 'ckpt', f'best_psnr_dec.pth')
            torch.save({'state_dict': enc.state_dict()}, path_save_enc)
            # torch.save({'state_dict': latent.state_dict()}, path_save_latent)
            torch.save({'state_dict': dec.state_dict()}, path_save_dec)
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

            path_save_enc = os.path.join(save_path, 'ckpt', f'best_ssim_enc.pth')
            # path_save_latent = os.path.join(save_path, 'ckpt', f'best_ssim_latent.pth')
            path_save_dec = os.path.join(save_path, 'ckpt', f'best_ssim_dec.pth')
            torch.save({'state_dict': enc.state_dict()}, path_save_enc)
            # torch.save({'state_dict': latent.state_dict()}, path_save_latent)
            torch.save({'state_dict': dec.state_dict()}, path_save_dec)
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
    name = 'ED'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.Tensor([0.1, 1.0]).to(device)
    criterion = nn.MSELoss().to(device)
    # 'interval': 2, 'target_num': 1, cfg.name = 'BidirecRNN'
    in_shape = [cfg.interval, cfg.channel, cfg.height, cfg.weight]
    seed = int(time.time())
    save_path = f'./Logs/{name}_{seed}'  # 保存模型
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
        enc = Encoder(1, 16, 4, 3, act_inplace=False).to(device)
        dec = Decoder(16, 1, 4, 3, act_inplace=False).to(device)
        checkpoint_enc = torch.load('./Logs/ckpt/best_ssim_enc.pth', map_location=device)
        checkpoint_dec = torch.load('./Logs/ckpt/best_ssim_dec.pth', map_location=device)
        enc.load_state_dict(checkpoint_enc['state_dict'])
        dec.load_state_dict(checkpoint_dec['state_dict'])
        # para = list(enc.parameters()) + list(dec.parameters())
        latent = ODENet(16, True, t).to(device)
        para = list(latent.parameters())
        # para = list(enc.parameters()) + list(latent.parameters()) + list(dec.parameters())
        optimizer_uni = torch.optim.SGD(para, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler_uni = OneCycleLR(optimizer_uni, max_lr=cfg.learning_rate, steps_per_epoch=len(train_dataloader),
                                   epochs=cfg.epochs)
        with open(record_file, 'a') as result:
            for e in range(cfg.epochs):
                train_unidirec(e + 1, record, result, train_dataloader)
                # 在test数据上进行测试
                test_unidirec(e + 1, record, result, test_dataloader)

            # 保存最后一次模型
            path_save_enc = os.path.join(save_path, 'ckpt', f'last_enc.pth')
            path_save_latent = os.path.join(save_path, 'ckpt', f'last.pth')
            path_save_dec = os.path.join(save_path, 'ckpt', f'last_dec.pth')
            torch.save({'state_dict': enc.state_dict()}, path_save_enc)
            # torch.save({'state_dict': latent.state_dict()}, path_save_latent)
            torch.save({'state_dict': dec.state_dict()}, path_save_dec)
            print('Accomplished!')
            result.write('Accomplished!')
    elif cfg.name == 'BidirecLSTM':
        # bidirection
        simvp_model_forward = SimVP_Model(in_shape=in_shape, t=t).to(device)
        simvp_model_reverse = SimVP_Model(in_shape=in_shape, t=t).to(device)
        param_bi = (list(simvp_model_forward.parameters()) + list(simvp_model_reverse.parameters()))
        optimizer_bi = torch.optim.Adam(param_bi, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler_bi = OneCycleLR(optimizer_bi, max_lr=cfg.learning_rate, steps_per_epoch=len(train_dataloader),
                                   epochs=cfg.epochs)
        # 定义全局最优pnsr和ssim，以及对应的epoch
        with open(record_file, 'a') as result:
            for e in range(cfg.epochs):
                train_bidirec(e + 1, record, result, train_dataloader)
                # 在val数据上进行验证
                test_bidirec(e + 1, record, result, test_dataloader)

            # 保存最后一次的模型
            path_save_forward = os.path.join(save_path, 'ckpt', f'last_simvp_forward.pth')
            path_save_reverse = os.path.join(save_path, 'ckpt', f'last_simvp_reverse.pth')

            torch.save({'state_dict': simvp_model_forward.state_dict()}, path_save_forward)
            torch.save({'state_dict': simvp_model_reverse.state_dict()}, path_save_reverse)

            print('Accomplished!')
            result.write('Accomplished!')
