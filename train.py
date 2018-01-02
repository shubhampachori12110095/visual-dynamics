from __future__ import print_function

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import MotionDataset
from networks import VDNet
from utils.torch import Logger, to_var

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)

    # dataset
    parser.add_argument('--data_path', default = '/data/vision/billf/motionTransfer/data/toy/shapes/')
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 8, type = int)

    # optimization
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--grad_clip', default = 1., type = float)

    # training
    parser.add_argument('--epochs', default = 128, type = int)
    parser.add_argument('--snapshot', default = 1, type = int)
    parser.add_argument('--gpu', default = '0')

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # cuda devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # datasets & loaders
    data, loaders = {}, {}
    for split in ['train', 'test']:
        data[split] = MotionDataset(data_path = args.data_path, split = split)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0} + {1}'.format(len(data['train']), len(data['test'])))

    # model
    model = VDNet()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    print('==> optimizer loaded')

    # experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = False)

    # logger
    logger = Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))

    # snapshot
    if args.resume is not None:
        if os.path.isfile(args.resume):
            snapshot = torch.load(args.resume)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            optimizer.load_state_dict(snapshot['optimizer'])
            print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
    else:
        epoch = 0

    # training
    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])

        for inputs, targets in tqdm(loaders['train'], desc = 'epoch {0}'.format(epoch + 1)):
            inputs, targets = to_var(inputs), to_var(targets)

            # forward
            model.forward(inputs)
