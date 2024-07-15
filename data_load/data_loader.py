import gzip
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from bidirectional_lstm.config_para import cfg


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_mnist(root, data_name='mnist'):  # return (60000, 28, 28) MNIST dataset
    # Load MNIST dataset for generating training data.
    file_map = {
        'mnist': 'moving_mnist/train-images-idx3-ubyte.gz',
        'fmnist': 'moving_fmnist/train-images-idx3-ubyte.gz',
        'mnist_cifar': 'moving_mnist/train-images-idx3-ubyte.gz',
    }
    path = os.path.join(root, file_map[data_name])
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root, data_name='mnist'):  # return (20, 10000, 64, 64, 1) moving MNIST dataset
    # Load the fixed dataset
    file_map = {
        'mnist': 'moving_mnist/mnist_test_seq.npy',
        'fmnist': 'moving_fmnist/fmnist_test_seq.npy',
        'mnist_cifar': 'moving_mnist/mnist_cifar_test_seq.npy',
    }
    path = os.path.join(root, file_map[data_name])
    dataset = np.load(path)
    if 'cifar' not in data_name:
        dataset = dataset[..., np.newaxis]
    return dataset


# dataset for interpolating
class subDataset(Dataset):
    def __init__(self, root_path, interval, target_num, channel, image_size, isTrain, use_augment):
        super(subDataset, self).__init__()
        self.root_path = root_path
        self.interval = interval
        self.target_num = target_num
        self.seq_len_before = (interval * 2 - 1) * (target_num + 1) + 1
        self.seq_len = interval * 2 + 1
        self.start_index = (interval - 1) * (target_num + 1) + 1
        self.end_index = self.start_index + target_num
        self.isTrain = isTrain
        if self.isTrain:
            self.data = load_mnist(self.root_path)
        else:
            self.data = load_fixed_set(self.root_path)
        self.image_size = image_size
        self.image_size_ = 28
        self.step_size = 0.2
        self.use_augment = use_augment
        self.num_objects = 2
        self.channel = channel

    def __len__(self):
        if self.isTrain:
            return len(self.data)
        else:
            return self.data.shape[1]

    def __getitem__(self, idx):
        if self.isTrain:
            # Generate data on the fly
            moving_sample = self.generate_moving_mnist(self.num_objects)
            # Generate the training sequence: training data and corresponding labels
            images_interval = moving_sample[::self.target_num + 1]
            images_target = moving_sample[self.start_index:self.end_index]
            images = np.concatenate([images_interval, images_target], axis=0)

        else:
            moving_sample = self.data[:self.seq_len_before, idx, ...]
            # # 创建 1 行 11 列的子图
            # fig, axes = plt.subplots(1, 11, figsize=(18, 4))
            #
            # # 遍历子图并绘制数据
            # for i, ax in enumerate(axes):
            #     ax.imshow(moving_sample[i], cmap='gray')
            #     ax.set_title(f'{i + 1}')
            #
            # # 调整间距并显示图形
            # plt.subplots_adjust(wspace=0.4)
            # plt.show()
            # exit()
            images_interval = moving_sample[::self.target_num + 1]
            images_target = moving_sample[self.start_index:self.end_index]
            images = np.concatenate([images_interval, images_target], axis=0)

        if self.channel == 1:
            images = images[:self.seq_len].reshape(
                (self.seq_len, self.image_size, self.channel, self.image_size, self.channel)).transpose(
                0, 2, 4, 1, 3).reshape((self.seq_len, self.channel, self.image_size, self.image_size))
        else:
            images = images[:self.seq_len].reshape(
                (self.seq_len, self.image_size, self.channel, self.image_size, self.channel)).transpose(
                0, 2, 4, 1, 3).reshape((self.seq_len, self.channel * self.channel, self.image_size, self.image_size))
        images = torch.from_numpy(images / 255.0).contiguous().float()

        # unique_values = np.unique(images)



        input = images[: self.interval * 2]
        if self.target_num > 0:
            output = images[-self.target_num:]
        else:
            output = []

        if self.use_augment:
            imgs = self._augment_seq(torch.cat([input, output], dim=0), crop_scale=0.94)
            input = imgs[:self.interval * 2, ...]
            output = imgs[-self.target_num:, ...]

        return input, output
        # print(idx)

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size - self.image_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi

        v_ys = [np.sin(theta)] * seq_length
        v_xs = [np.cos(theta)] * seq_length

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        bounce_x = 1
        bounce_y = 1
        for i, v_x, v_y in zip(range(seq_length), v_xs, v_ys):
            # Take a step along velocity.
            y += bounce_y * v_y * self.step_size
            x += bounce_x * v_x * self.step_size

            # Bounce off edges.
            if x <= 0:
                x = 0
                # v_x = -v_x
                bounce_x = -bounce_x
            if x >= 1.0:
                x = 1.0
                # v_x = -v_x
                bounce_x = -bounce_x
            if y <= 0:
                y = 0
                # v_y = -v_y
                bounce_y = -bounce_y
            if y >= 1.0:
                y = 1.0
                # v_y = -v_y
                bounce_y = -bounce_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_objects):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        samples = np.zeros((self.seq_len_before, self.image_size,
                            self.image_size), dtype=np.float32)
        # Trajectory
        for n in range(num_objects):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.seq_len_before)
            ind = random.randint(0, self.data.shape[0] - 1)
            digit_image = self.data[ind].copy()

            for i in range(self.seq_len_before):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size_
                right = left + self.image_size_
                # Draw digit
                samples[i, top:bottom, left:right] = np.maximum(samples[i, top:bottom, left:right], digit_image)

        moving_sample = samples[..., np.newaxis]
        return moving_sample

    def _augment_seq(self, imgs, crop_scale=0.94):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 1, 64, 64]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x + h, y:y + w]
        # Random Flip
        if random.randint(-2, 1):
            imgs = torch.flip(imgs, dims=(2, 3))  # rotation 180
        elif random.randint(-2, 1):
            imgs = torch.flip(imgs, dims=(2,))  # vertical flip
        elif random.randint(-2, 1):
            imgs = torch.flip(imgs, dims=(3,))  # horizontal flip
        return imgs


if __name__ == '__main__':
    subdataset = subDataset(root_path='./datasets', interval=3, target_num=1, channel=1,
                            image_size=cfg.image_size,
                            isTrain=False, use_augment=False)
    index = random.randint(0, 9999)
    print(index)
    it = subdataset.__getitem__(index)
