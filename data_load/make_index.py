import numpy as np

from config_para import cfg

"""该文件获得train.txt val.txt test.txt"""


def data_loader(path):
    # load moving moving_mnist data, data shape = [time steps, batch size, width, height] = [20, batch_size, 64, 64]
    # B S H W -> S B H W
    data = np.load(path)
    data_trans = data.transpose(1, 0, 2, 3)
    return data_trans


def make_idx_txt(Path, IdList, interval=2, T=20):
    f = open(Path, 'w')
    for i in IdList:
        len_seq = int(T - interval * 2)
        for t in range(len_seq):
            f.write(
                '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i, t, t + 1, t + 2, t + 3, t + 4, t + 6, t + 7, t + 8,
                                                               t + 9, t + 10, t + 5))


if __name__ == '__main__':
    data = data_loader('./data_load/mnist_test_seq.npy')
    # train_txt
    n_train = int(cfg.total_num * cfg.train_share)
    train_id_list = [i for i in range(n_train)]
    make_idx_txt(cfg.train_path, train_id_list, cfg.interval, cfg.T)
    np.save(f'./data/mnist_train.npy', data[:n_train, :, :, :])

    # val_txt
    n_val = int(cfg.total_num * cfg.val_share)
    val_id_list = [i for i in range(n_val)]
    make_idx_txt(cfg.val_path, val_id_list, cfg.interval, cfg.T)
    np.save(f'./data/mnist_val.npy', data[n_train:n_train + n_val, :, :, :])

    # test_txt
    n_test = int(cfg.total_num * cfg.test_share)
    test_id_list = [i for i in range(n_test)]
    make_idx_txt(cfg.test_path, test_id_list, cfg.interval, cfg.T)
    np.save(f'./data/mnist_test.npy', data[n_train + n_val:n_train + n_val + n_test, :, :, :])
