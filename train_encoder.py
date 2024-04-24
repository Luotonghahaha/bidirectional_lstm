# from data_load.data_pro_2 import *
import torch
from models.encoder_model import Encoder, Decoder, ODEfunc
from torch.utils.data import DataLoader
from evaluation.ssim import ssim
from evaluation.psnr import psnr
from evaluation.ie import ie
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_load.data_loader import subDataset_i
from config_para import cfg
import os
import time
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from torchvision import transforms as T

from skimage.metrics import peak_signal_noise_ratio as psnr2
from skimage.metrics import structural_similarity as ssim2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# get three datasets respectively
dataset_train = subDataset_i(data_npy=cfg.train_npy_path, isTrain=True)
dataset_val = subDataset_i(data_npy=cfg.val_npy_path, isTrain=False)
dataset_test = subDataset_i(data_npy=cfg.test_npy_path, isTrain=False)
train_dataloader = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
val_dataloader = DataLoader(dataset=dataset_val, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, drop_last=True)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, drop_last=True)
print('All data is ready!')

encoder = Encoder(4096, 1024, 256)
decoder = Decoder(256, 1024, 4096)
ode_fun = ODEfunc(256)
global_t = torch.FloatTensor([0, 1])

criterion = nn.MSELoss().to(device)
params = (list(encoder.parameters()) + list(ode_fun.parameters()) + list(decoder.parameters()))
optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

def lr_scheduler(opt, epoch, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if (epoch + 1) % lr_decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return opt


def train(epoch, record, result, train_dataloader):
    loss_train = 0.0
    ssim_train = 0.0
    psnr_train = 0.0
    ie_train = 0.0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x_train = data.to(device)
        encoder_x = encoder(x_train)    # 作为ode的输入
        ode_x = odeint(ode_fun, encoder_x, global_t, method='euler')[1]
        decoder_x = decoder(ode_x)
        decoder_x = decoder_x.reshape(x_train.shape)
        loss = criterion(decoder_x, x_train)
        ssim_batch = ssim(decoder_x, x_train)
        psnr_batch = psnr(loss)
        # print(psnr_batch)
        # exit()
        # ie_batch = ie(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        ssim_train += ssim_batch.item()
        psnr_train += psnr_batch.item()

        # ie_train += ie_batch
    record.add_scalar('Loss_Train', loss_train, epoch)
    record.add_scalar('SSIM_Train', ssim_train, epoch)
    record.add_scalar('PSNR_Train', psnr_train, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    lr_scheduler(optimizer, epoch)
    print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train / (i + 1):.4f}, SSIM: {ssim_train / (i + 1):.2f}, PSNR: {psnr_train / (i + 1):.2f}')
    result.write(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train / (i + 1):.4f}, SSIM: {ssim_train / (i + 1):.2f}, PSNR: {psnr_train / (i + 1):.2f}\n')


def test(epoch, record, result, test_dataloader):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        best_psnr = 0.0
        best_ssim = 0.0
        loss_val = 0.0
        ssim_val = 0.0
        psnr_val = 0.0
        # ie_test = 0.0

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x_test= data.to(device)
            encoder_x = encoder(x_test)
            ode_x = odeint(ode_fun, encoder_x, global_t, method='euler')[1]
            decoder_x = decoder(ode_x)
            decoder_x = decoder_x.reshape(x_test.shape)
            loss = criterion(decoder_x, x_test)
            ssim_batch = ssim(decoder_x, x_test)
            psnr_batch = psnr(loss)

            # print(psnr_batch)
            # exit()
            # ie_batch = ie(y)
            loss_val += loss
            ssim_val += ssim_batch
            psnr_val += psnr_batch
            # ie_test += ie_batch

        record.add_scalar('Loss_Test', loss_val, epoch)
        record.add_scalar('SSIM_Test', ssim_val, epoch)
        record.add_scalar('PSNR_Test', psnr_val, epoch)
        # record.add_scalar('PSNR_Test', ie_test, epoch)
        # return loss, ssim_test, psnr_test, ie_test
        loss_val = loss_val / (i + 1)
        ssim_val = ssim_val / (i + 1)
        psnr_val = psnr_val / (i + 1)
        print(f'Test on Epoch{epoch}/{cfg.epochs}! Loss: {loss_val:.4f}, SSIM_val: {ssim_val:.2f}, PSNR_val: {psnr_val:.2f}')
        result.write(f'Test on Epoch{epoch}/{cfg.epochs}! Loss: {loss_val:.4f}, SSIM_val: {ssim_val:.2f}, PSNR_val: {psnr_val:.2f}\n')
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.2f}, ssim:{ssim_val:.2f}')
            result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.2f}, ssim:{ssim_val:.2f}\n')
            path_encoder = os.path.join(save_path, 'ckpt', f'best_psnr_encoder.pth')
            path_decoder = os.path.join(save_path, 'ckpt', f'best_psnr_decoder.pth')
            torch.save({'state_dict': encoder.state_dict()}, path_encoder)
            torch.save({'state_dict': decoder.state_dict()}, path_decoder)

        if ssim_val > best_ssim:
            best_ssim = ssim_val
            best_epoch = epoch
            # path = os.path.join(save_path, 'ckpt',
            #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
            print(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_val:.2f}, best_ssim:{best_ssim:.2f}')
            result.write(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_val:.2f}, best_ssim:{best_ssim:.2f}\n')
            path_encoder = os.path.join(save_path, 'ckpt', f'best_ssim_encoder.pth')
            path_decoder = os.path.join(save_path, 'ckpt', f'best_ssim_decoder.pth')
            torch.save({'state_dict': encoder.state_dict()}, path_encoder)
            torch.save({'state_dict': decoder.state_dict()}, path_decoder)



if __name__ == '__main__':
    name = 'Encoder_Decoder'
    seed = int(time.time())
    save_path = f'./Logs/{name}_{seed}'     # 保存模型

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'ckpt'))
    record = SummaryWriter(log_dir=save_path)
    # result = open(result_save_path, 'w')
    record_file = save_path + '/output.txt'
    result = open(record_file, 'a')
    for e in range(cfg.epochs):
        train(e + 1, record, result, train_dataloader)
        # loss, ssim_eval, psnr_eval, ie_eval = test(e, record, val_dataloader)
        test(e + 1, record, result, val_dataloader)
    # 保存最后一次的模型
    print('Test on test dataset:')
    result.write('Test on test dataset:\n')
    path_encoder = os.path.join(save_path, 'ckpt', f'last_training_model_encoder.pth')
    path_decoder = os.path.join(save_path, 'ckpt', f'last_training_model_decoder.pth')
    torch.save({'state_dict': encoder.state_dict()}, path_encoder)
    torch.save({'state_dict': decoder.state_dict()}, path_decoder)

    test(cfg.epochs, record, result, test_dataloader)
    print('Accomplished!')
    result.write('Accomplished!')
