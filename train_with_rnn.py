import torch
import numpy as np
from models.models import Encoder, Decoder, RNNModel
from torch.utils.data import DataLoader
from evaluation.ssim import ssim
from evaluation.psnr import psnr
from evaluation.ie import ie
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_load.data_loader import subDataset
from config_para import cfg
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms as T

# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim


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

encoder = Encoder().to(device)
latent_model = RNNModel(cfg.hidden_rnn, cfg.hidden_rnn).to(device)
decoder = Decoder().to(device)
h_0 = torch.zeros(1, cfg.batch_size, cfg.hidden_rnn).to(device)

# model = UnetModel().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.MSELoss().to(device)
params = (list(encoder.parameters()) + list(latent_model.parameters()) + list(decoder.parameters()))
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


# 训练单向RNN: 由前后各interval帧生成中间1帧
def train_unidirec(epoch, record, result, train_dataloader):
    loss_train = 0.0
    ssim_train = []
    psnr_train = []
    ie_train = 0.0

    encoder.train()
    decoder.train()
    latent_model.train()
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x_train, y_train = data[0].to(device), data[1].to(device)
        # x_input = x_train.unsqueeze(2).transpose(0, 1)
        hidden_x = encoder(x_train.unsqueeze(2).transpose(0, 1))
        hidden_y, _ = latent_model(hidden_x.view(cfg.interval * 2, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
        # print(hidden_y.shape)
        # exit()
        # decoder_y = hidden_y.view(cfg.batch_size, cfg.hidden_rnn, 1, 1)
        pred = decoder(hidden_y.view(cfg.batch_size, cfg.hidden_rnn, 1, 1))
        # print(pred.shape)
        # exit()

        loss = criterion(pred, y_train)
        ssim_batch = ssim(pred, y_train)
        psnr_batch = psnr(loss)
        # print(psnr_batch)
        # exit()
        # ie_batch = ie(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        ssim_train.append(ssim(pred, y_train))
        psnr_train.append(psnr(pred, y_train))

        # ie_train += ie_batch
    ssim_train = np.concatenate(ssim_train, axis=0).mean()
    psnr_train = np.concatenate(psnr_train, axis=0).mean()
    record.add_scalar('Loss_Train', loss_train, epoch)
    record.add_scalar('SSIM_Train', ssim_train, epoch)
    record.add_scalar('PSNR_Train', psnr_train, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    lr_scheduler(optimizer, epoch)
    print(
        f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train / (i + 1):.4f}, SSIM: {ssim_train / (i + 1):.4f}, PSNR: {psnr_train / (i + 1):.4f}')
    result.write(
        f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train / (i + 1):.4f}, SSIM: {ssim_train / (i + 1):.4f}, PSNR: {psnr_train / (i + 1):.4f}\n')


# 训练双向RNN: 分别由前后各interval帧生成中间1帧,然后将两个结果加权组合
def train_bidirec(epoch, record, result, train_dataloader):
    loss_train = 0.0
    ssim_train = []
    psnr_train = []
    ie_train = 0.0

    encoder.train()
    decoder.train()
    latent_model.train()
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x_train, y_train = data[0].to(device), data[1].to(device)
        hidden_x = encoder(x_train.unsqueeze(2).transpose(0, 1))
        # hidden_y, _ = latent_model(hidden_x)
        forward_y, _ = latent_model(hidden_x[0: 2].view(cfg.interval, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
        reverse_y, _ = latent_model(torch.flip(hidden_x[2:], dims=[0]).view(cfg.interval, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
        hidden_y = cfg.epsilon * forward_y + (1 - cfg.epsilon) * reverse_y

        pred = decoder(hidden_y.view(cfg.batch_size, cfg.hidden_rnn, 1, 1))


        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        ssim_train.append(ssim(pred, y_train))
        psnr_train.append(psnr(pred, y_train))

    ssim_train = np.concatenate(ssim_train, axis=0).mean()
    psnr_train= np.concatenate(psnr_train, axis=0).mean()
    record.add_scalar('Loss_Train', loss_train, epoch)
    record.add_scalar('SSIM_Train', ssim_train, epoch)
    record.add_scalar('PSNR_Train', psnr_train, epoch)
    # record.add_scalar('IE_Train', ie_train, epoch)
    # print(f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_sum / (i + 1):.4f},\
    #         SSIM: {ssim_sum / (i + 1):.2f}, PSNR: {psnr_sum / (i + 1):.2f},\
    #         IE: {psnr_sum / (i + 1):.2f}')
    lr_scheduler(optimizer, epoch)
    print(
            f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train / (i + 1):.4f}, SSIM: {ssim_train :.4f}, PSNR: {psnr_train :.4f}')
    result.write(
            f'Epoch {epoch}/{cfg.epochs}, Loss: {loss_train / (i + 1):.4f}, SSIM: {ssim_train :.4f}, PSNR: {psnr_train :.4f}')


# 测试单向RNN: 分别由前后各interval帧生成中间1帧,然后将两个结果加权组合
def test_unidirec(epoch, record, result, test_dataloader):
    global best_psnr, best_ssim, best_epoch
    encoder.eval()
    latent_model.eval()
    decoder.eval()

    with torch.no_grad():

        loss_val = 0.0
        psnr_val = []
        ssim_val = []
        # ie_test = 0.0

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x_test, y_test = data[0].to(device), data[1].to(device)

            hidden_x = encoder(x_test.unsqueeze(2).transpose(0, 1))
            hidden_y, _ = latent_model(hidden_x.view(cfg.interval * 2, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
            # print(hidden_y.shape)
            # exit()
            # decoder_y = hidden_y.view(cfg.batch_size, cfg.hidden_rnn, 1, 1)
            y = decoder(hidden_y.view(cfg.batch_size, cfg.hidden_rnn, 1, 1))

            loss = criterion(y, y_test)
            ssim_batch = ssim(y, y_test)
            psnr_batch = psnr(y, y_test)

            loss_val += loss
            ssim_val.append(ssim_batch)
            psnr_val.append(psnr_batch)

        # record.add_scalar('PSNR_Test', ie_test, epoch)
        # return loss, ssim_test, psnr_test, ie_test
        loss_val = loss_val / (i + 1)
        psnr_val = np.concatenate(psnr_val, axis=0).mean()
        ssim_val = np.concatenate(ssim_val, axis=0).mean()
        record.add_scalar('Loss_Test', loss_val, epoch)
        record.add_scalar('SSIM_Test', ssim_val, epoch)
        record.add_scalar('PSNR_Test', psnr_val, epoch)
        print(
            f'Test on Epoch{epoch}/{cfg.epochs}! Loss: {loss_val:.4f}, SSIM_val: {ssim_val:.4f}, PSNR_val: {psnr_val:.4f}')
        result.write(
            f'Test on Epoch{epoch}/{cfg.epochs}! Loss: {loss_val:.4f}, SSIM_val: {ssim_val:.4f}, PSNR_val: {psnr_val:.4f}\n')
    if psnr_val > best_psnr:
        best_psnr = psnr_val
        best_epoch = epoch
        # path = os.path.join(save_path, 'ckpt',
        #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
        print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_val:.4f}')
        result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_val:.4f}\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'best_psnr_encoder.pth')
        path_rnn = os.path.join(save_path, 'ckpt', f'best_psnr_rnn.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'best_psnr_decoder.pth')
        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': latent_model.state_dict()}, path_rnn)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
    else:
        print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}')
        result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}\n')

    if ssim_val > best_ssim:
        best_ssim = ssim_val
        best_epoch = epoch
        # path = os.path.join(save_path, 'ckpt',
        #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
        print(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_val:.4f}, best_ssim:{best_ssim:.4f}')
        result.write(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_val:.4f}, best_ssim:{best_ssim:.4f}\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'best_ssim_encoder.pth')
        path_rnn = os.path.join(save_path, 'ckpt', f'best_ssim_rnn.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'best_ssim_decoder.pth')
        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': latent_model.state_dict()}, path_rnn)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
    else:
        print(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}')
        result.write(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}\n')


# 测试双向RNN: 由前后各interval帧生成中间1帧，对他们的结果进行加权组合作为中间帧
def test_bidirec(epoch, record, result, test_dataloader):
    global best_psnr, best_ssim, best_epoch
    encoder.eval()
    latent_model.eval()
    decoder.eval()

    loss_val = 0.0
    psnr_val = []
    ssim_val = []
    with torch.no_grad():
        # ie_test = 0.0

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x_test, y_test = data[0].to(device), data[1].to(device)
            hidden_x = encoder(x_test.unsqueeze(2).transpose(0, 1))
            # hidden_y, _ = latent_model(hidden_x)
            forward_y, _ = latent_model(hidden_x[0: 2].view(cfg.interval, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
            reverse_y, _ = latent_model(torch.flip(hidden_x[2:], dims=[0]).view(cfg.interval, cfg.batch_size, cfg.hidden_rnn * 1 * 1), h_0)
            hidden_y = cfg.epsilon * forward_y + (1 - cfg.epsilon) * reverse_y

            y = decoder(hidden_y.view(cfg.batch_size, cfg.hidden_rnn, 1, 1))

            loss = criterion(y, y_test)
            ssim_batch = ssim(y, y_test)
            psnr_batch = psnr(y, y_test)

            loss_val += loss
            ssim_val.append(ssim_batch)
            psnr_val.append(psnr_batch)

        # record.add_scalar('PSNR_Test', ie_test, epoch)
        # return loss, ssim_test, psnr_test, ie_test
        loss_val = loss_val / (i + 1)
        ssim_val = np.concatenate(ssim_val, axis=0).mean()
        psnr_val = np.concatenate(psnr_val, axis=0).mean()
        record.add_scalar('Loss_Test', loss_val, epoch)
        record.add_scalar('SSIM_Test', ssim_val, epoch)
        record.add_scalar('PSNR_Test', psnr_val, epoch)
        print(
            f'Test on Epoch{epoch}/{cfg.epochs}! Loss: {loss_val:.4f}, SSIM_val: {ssim_val:.4f}, PSNR_val: {psnr_val:.4f}')
        result.write(
            f'Test on Epoch{epoch}/{cfg.epochs}! Loss: {loss_val:.4f}, SSIM_val: {ssim_val:.4f}, PSNR_val: {psnr_val:.4f}\n')
    if psnr_val > best_psnr:
        best_psnr = psnr_val
        best_epoch = epoch
        # path = os.path.join(save_path, 'ckpt',
        #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
        print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_val:.4f}')
        result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}, ssim:{ssim_val:.4f}\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'best_psnr_encoder.pth')
        path_rnn = os.path.join(save_path, 'ckpt', f'best_psnr_rnn.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'best_psnr_decoder.pth')
        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': latent_model.state_dict()}, path_rnn)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
    else:
        print(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}')
        result.write(f'best_psnr_epoch: {best_epoch}, best_psnr:{best_psnr:.4f}\n')

    if ssim_val > best_ssim:
        best_ssim = ssim_val
        best_epoch = epoch
        # path = os.path.join(save_path, 'ckpt',
        #                     f'Epoch{e}-psnr{psnr_eval}-ssim{ssim_eval}-ie{ie_eval}-Loss{loss}.pth')
        print(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_val:.4f}, best_ssim:{best_ssim:.4f}')
        result.write(f'best_ssim_epoch: {best_epoch}, psnr:{psnr_val:.4f}, best_ssim:{best_ssim:.4f}\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'best_ssim_encoder.pth')
        path_rnn = os.path.join(save_path, 'ckpt', f'best_ssim_rnn.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'best_ssim_decoder.pth')
        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': latent_model.state_dict()}, path_rnn)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
    else:
        print(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}')
        result.write(f'best_ssim_epoch: {best_epoch}, best_ssim:{best_ssim:.4f}\n')


if __name__ == '__main__':
    name = 'BidirecRNN'
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
            train_bidirec(e + 1, record, result, train_dataloader)
            # loss, ssim_eval, psnr_eval, ie_eval = test(e, record, val_dataloader)
            test_bidirec(e + 1, record, result, val_dataloader)
        # 保存最后一次的模型
        print('Test on test dataset:')
        result.write('Test on test dataset:\n')
        path_encoder = os.path.join(save_path, 'ckpt', f'last_encoder.pth')
        path_rnn = os.path.join(save_path, 'ckpt', f'last_rnn.pth')
        path_decoder = os.path.join(save_path, 'ckpt', f'last_decoder.pth')
        torch.save({'state_dict': encoder.state_dict()}, path_encoder)
        torch.save({'state_dict': latent_model.state_dict()}, path_rnn)
        torch.save({'state_dict': decoder.state_dict()}, path_decoder)
        train_bidirec(cfg.epochs, record, result, test_dataloader)
        print('Accomplished!')
        result.write('Accomplished!')
