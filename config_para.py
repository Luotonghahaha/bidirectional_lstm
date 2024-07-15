# moving_mnist包含10000张contains 10,000 sequences
# each of length 20 showing 2 digits moving in a 64 x 64 frame
import torch
from easydict import EasyDict
para = {
    'data_path': './datasets/moving_mnist/mnist_test_seq.npy',
    'root_path': './datasets',
    'total_num': 10000,
    'train_share': 0.8,
    'val_share': 0.1,
    'test_share': 0.1,
    'train_npy_path': './data/mnist_train.npy',
    'val_npy_path': './data/mnist_val.npy',
    'test_npy_path': './data/mnist_test.npy',
    'train_path': './data/train_5.txt',
    'val_path': './data/val_5.txt',
    'test_path': './data/test_5.txt',
    'data_dir': './data',
    'save_path': './out',
    'batch_size': 64,
    'interval': 3,
    'target_num': 1,
    'channel': 1,
    'image_size': 64,
    'height': 64,
    'weight': 64,
    't': torch.Tensor([0.0, 1.0]),
    'epochs': 300,
    'T': 20,
    'weight_decay': 1e-4,
    'learning_rate': 1e-3,
    'hidden_rnn': 512,
    'epsilon': 0.5,
    'name': 'UnidirecLSTM',
}
cfg = EasyDict(para)
