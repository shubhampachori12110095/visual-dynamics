from __future__ import print_function
import argparse
from data import MotionDataset
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)

    # dataset
    parser.add_argument('--data_path', default = '/data/vision/billf/motionTransfer/data/toy/shapes/')

    # training
    parser.add_argument('--epochs', default = 128, type = int)
    parser.add_argument('--batch', default = 128, type = int)
    parser.add_argument('--snapshot', default = 1, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--gpu', default = '0')

    # parse arguments
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
    model = None

    # optimizer
    optimizer = None

    # load the snapshot of model and optimizer
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

    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])

        for inputs, targets in tqdm(loaders['train'], desc = 'epoch {0}'.format(epoch + 1)):
            print(inputs, targets)
