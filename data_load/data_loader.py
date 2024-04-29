import random
from config_para import cfg
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = resized_img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')


# dataset for interpolating
class subDataset(Dataset):
    def __init__(self, data_txt, data_npy, isTrain):
        super(subDataset, self).__init__()
        self.isTrain = isTrain
        self.data_npy = np.load(data_npy)
        self.data = []

        # # 获取所有数据集
        # datatemp = open(data_txt, 'r').readlines()
        # for line in datatemp:
        #     self.data.append([int(i) for i in line.strip().split(',')])
        #     # print([int(i) for i in line.strip().split(',')])
        #     # print(self.data)

        # 用500个样本做测试
        datatemp = open(data_txt, 'r')
        for i in range(500):
            self.data.append([int(i) for i in datatemp.readline().strip().split(',')])

        self.transform_rotation_180 = T.Compose([
            T.ToPILImage(),
            T.RandomApply([T.RandomResizedCrop(size=(64, 64))], p=0.3),
            T.RandomHorizontalFlip(p=1),
            T.RandomVerticalFlip(p=1),
            T.ToTensor(),
        ])
        self.transform_horizontal_flip = T.Compose([
            T.ToPILImage(),
            T.RandomApply([T.RandomResizedCrop(size=(64, 64))], p=0.3),
            T.RandomHorizontalFlip(p=1),
            T.Resize(size=(64, 64)),
            T.ToTensor(),
        ])
        self.transform_vertical_flip = T.Compose([
            T.ToPILImage(),
            T.RandomApply([T.RandomResizedCrop(size=(64, 64))], p=0.3),
            T.RandomVerticalFlip(p=1),
            T.Resize(size=(64, 64)),
            T.ToTensor(),
        ])
        self.transform_test = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        txtline = self.data[idx]
        video = self.data_npy[txtline[0], txtline[1:], :, :]  # 前边interval * 2列是data,后面的列是target
        seq_len = video.shape[0]
        data_temp = torch.zeros(video.shape)  # [interval * 2 + cfg.target_num , H, W]

        if self.isTrain:
            if random.randint(-2, 1):
                for i in range(seq_len):
                    data_temp[i] = self.transform_rotation_180(video[i])    # rotation 180
            elif random.randint(-2, 1):
                for i in range(seq_len):
                    data_temp[i] = self.transform_horizontal_flip(video[i])    # horizontal flip
            elif random.randint(-2, 1):
                for i in range(seq_len):
                    data_temp[i] = self.transform_vertical_flip(video[i])  # vertical flip

        else:
            for i in range(seq_len):
                data_temp[i] = self.transform_test(video[i])
            # data_temp[-1] = self.test_target_transform(np.expand_dims(video[-1], axis=-1))

        data = data_temp[:cfg.interval*2, :, :].unsqueeze(1)
        target = data_temp[cfg.interval*2:, :, :].unsqueeze(1)

        return data, target


# demo dataset for generating video series
class subDataset_demo(Dataset):
    def __init__(self, sample_index, data_txt_path, data_npy_path, isTrain):
        super(subDataset_demo, self).__init__()
        self.sample_index = sample_index
        self.data_npy_path = data_npy_path
        self.isTrain = isTrain
        self.data_npy = np.load(self.data_npy_path)
        self.data_demo = []

        # 只返回一条视频序列的6个样本
        datatemp = open(data_txt_path, 'r')
        for line in datatemp:
            line_ = [int(i) for i in line.strip().split(',')]
            if line_[0] == sample_index:
                self.data_demo.append(line_)
        # print(self.data_demo)
        # self.data.append([int(i) for i in datatemp.readline().strip().split(',')])

        # self.flip_transform = T.Compose([
        #     # T.ToPILImage(),
        #     T.RandomHorizontalFlip(p=1),  # 水平翻转
        #     T.RandomVerticalFlip(p=1)  # 垂直翻转
        # ])
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.data_demo)

    def __getitem__(self, idx):
        # print(idx)
        txtline = self.data_demo[idx]
        if random.random() < 0.5:
            id_ = txtline[0]
            scaleing = random.random() + 1
            if round(max(txtline[1:]) * scaleing) <= 19:
                txtline = [round(i * scaleing) for i in txtline[1:]]
                txtline = [id_] + txtline
        video = self.data_npy[txtline[0], txtline[1:], :, :]  # 中间interval * 2列是data,最后一列是target

        data_temp = torch.zeros(video.shape)
        for i in range(video.shape[0]):
            data_temp[i] = self.transform(video[i])

        data = data_temp[:-1, :, :]
        target = data_temp[-1, :, :].unsqueeze(0)


        return data, target


# encoder_decoder dataset
class subDataset_i(Dataset):
    def __init__(self, data_npy, isTrain):
        super(subDataset_i, self).__init__()
        self.data_npy = np.load(data_npy)
        self.H = self.data_npy.shape[2]
        self.W = self.data_npy.shape[3]
        self.data = self.data_npy.reshape(-1, self.H, self.W)
        self.isTrain = isTrain
        self.train_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,))
        ])
        self.test_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.isTrain:
            sample = self.train_transform(self.data[idx])
        else:
            sample = self.test_transform(self.data[idx])
        return sample


if __name__ == '__main__':
    # subdataset = subDataset_i(data_npy='data/mnist_train.npy', isTrain=True)
    # len = subdataset.__len__()
    # it = subdataset.__getitem__(10)

    subdataset = subDataset(data_txt='data/train_5.txt', data_npy='data/mnist_train.npy',
                            isTrain=True)
    it = subdataset.__getitem__(0)

    # subdataset = subDataset_demo(12, data_txt_path='data/test_2.txt', data_npy_path='data/mnist_test.npy',
    #                              isTrain=False)
    # it = subdataset.__getitem__(111)
