import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms as T
import random
import time
from PIL import Image
from config_para import cfg


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
        # datatemp = open(data_txt, 'r').readlines()
        datatemp = open(data_txt, 'r')

        # # 获取所有数据集
        # for line in datatemp:
        #     self.data.append([int(i) for i in line.strip().split(',')])
        #     # print([int(i) for i in line.strip().split(',')])
        #     # print(self.data)

        # 用500个样本做测试
        for i in range(500):
            self.data.append([int(i) for i in datatemp.readline().strip().split(',')])

        self.train_data_transform = T.Compose([
            T.ToPILImage(),         # 将Tensor变量的数据转换成PIL图片数据
            # T.CenterCrop(size=(32, 32)),
            # T.RandomCrop(size=(32, 32)),
            # T.Resize(size=(64, 64)),
            # T.RandomAffine(degrees=0, translate=(0.1, 0.1)),        # 随机仿射变换
            # T.RandomRotation((-10, 10)),    # 表示在（-10，10）之间随机旋转
            # T.RandomHorizontalFlip(),   # 按随机概率进行水平翻转，默认概率0.5
            # T.RandomVerticalFlip(),     # 按随机概率进行垂直翻转，默认概率0.5
            T.ToTensor(),   # 将PIL Image或numpy形式的图片转换为Tensor，让PyTorch能够对其进行计算和处理，维度为(C,H,W),并对像素值进行归一化。
            # T.Normalize((0.1307,), (0.3081,)),         # 数据标准化
            # T.ToPILImage()
        ])

        # self.train_target_transform = T.Compose([
        #     T.ToPILImage(),
        #     T.RandomCrop(size=(32, 32)),
        #     T.Resize(size=(64, 64)),
        #     # T.CenterCrop(size=(32, 32)),
        #     T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        #     T.RandomRotation((-10, 10)),
        #     T.RandomHorizontalFlip(),
        #     T.RandomVerticalFlip(),
        #     T.ToTensor(),
        #     T.Normalize((0.1307,), (0.3081,)),         # 数据标准化
        # ])

        self.test_data_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,))
        ])

        # self.test_target_transform = T.Compose([
        #     T.ToPILImage(),
        #     T.ToTensor(),
        # ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        txtline = self.data[idx]
        # 随机变速
        video = self.data_npy[txtline[0], txtline[1:], :, :]  # 中间interval * 2列是data,最后一列是target

        data_temp = torch.zeros(video.shape)
        if self.isTrain:
            # seed = int(time.time())
            for i in range(video.shape[0]):
                data_temp[i] = self.train_data_transform(np.expand_dims(video[i], axis=-1))

        else:
            for i in range(video.shape[0] - 1):
                data_temp[i] = self.test_data_transform(np.expand_dims(video[i], axis=-1))
            # data_temp[-1] = self.test_target_transform(np.expand_dims(video[-1], axis=-1))

        data = data_temp[:-1, :, :].unsqueeze(1)
        target = data_temp[-1, :, :].unsqueeze(0).unsqueeze(1)
        # print(data.shape)
        # print(target.shape)
        # print(data.unique())
        # print(target.unique())
        # exit()
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

    subdataset = subDataset(data_txt='data/train_2.txt', data_npy='data/mnist_train.npy',
                                 isTrain=True)
    it = subdataset.__getitem__(0)

    # subdataset = subDataset_demo(12, data_txt_path='data/test_2.txt', data_npy_path='data/mnist_test.npy',
    #                              isTrain=False)
    # it = subdataset.__getitem__(111)
