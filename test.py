from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import utils
from data import MotionDataset
from networks import VDNet
from utils.torch import load_snapshot

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--gpu', default = '0')

    # dataset
    parser.add_argument('--data_path', default = '/data/vision/billf/motionTransfer/data/toy/3Shapes2_large/')
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 8, type = int)

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # cuda devices
    utils.set_cuda_devices(args.gpu)

    # datasets & loaders
    data, loaders = {}, {}
    for split in ['train', 'test']:
        data[split] = MotionDataset(data_path = args.data_path, split = split)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0} + {1}'.format(len(data['train']), len(data['test'])))

    # model
    model = VDNet().cuda()

    # experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = False)

    # load snapshot
    means, log_vars = load_snapshot(args.resume, model = model, returns = ['means', 'log_vars'])

    # testing
    model.train(False)

    X, Y = [], []
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            X.append(j + 1)
            Y.append(means[i][j])
    sns.set()
    sns.set_style('white')

    plt.plot(X, Y)
    plt.savefig('mean.png', bbox_inches = 'tight')
