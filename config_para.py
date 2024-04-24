# moving_mnist包含10000张contains 10,000 sequences
# each of length 20 showing 2 digits moving in a 64 x 64 frame
from easydict import EasyDict
para = {
    'data_path': './mnist_test_seq.npy',
    'train_npy_path': './data/mnist_train.npy',
    'val_npy_path': './data/mnist_val.npy',
    'test_npy_path': './data/mnist_test.npy',
    # 'train_path': './data/train_2.txt',
    # 'val_path': './data/val_2.txt',
    # 'test_path': './data/test_2.txt',
    'train_path': './data/train.txt',
    'val_path': './data/val.txt',
    'test_path': './data/test.txt',
    'data_dir': './data',
    'save_path': './out',
    'batch_size': 64,
    'epochs': 300,
    'interval': 2,
    'target_num': 1,
    'total_num': 10000,
    'train_share': 0.8,
    'val_share': 0.1,
    'test_share': 0.1,
    'T': 20,
    'weight_decay': 5e-5,
    'learning_rate': 1e-3,
    'hidden_rnn': 512,
    'epsilon': 0.5,
    'name': 'BidirecRNN'
}
cfg = EasyDict(para)
