from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import utils
from data import MotionDataset
from misc import visualize
from networks import VDNet
from utils.torch import load_snapshot, save_images, to_np, to_var

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

    # test path
    test_path = os.path.join('exp', args.exp, 'test')
    utils.shell.mkdir(test_path, clean = True)

    # load snapshot
    means, log_vars = load_snapshot(args.resume, model = model, returns = ['means', 'log_vars'])


    def show_dist(data, path):
        num_dists, num_dims = data.shape

        x, y = [], []
        for k in range(num_dims):
            x.extend([k] * num_dists)
            y.extend(data[:, k])

        plt.figure()
        plt.plot(x, y)
        plt.savefig(path, bbox_inches = 'tight')


    # show_dist(means, os.path.join(test_path, 'means.png'))
    # show_dist(log_vars, os.path.join(test_path, 'vars.png'))

    images_path = os.path.join(test_path, 'images')
    utils.shell.mkdir(images_path, clean = True)

    std = np.std(means, axis = 0)

    indices = np.argsort(std)

    dimensions = indices[-5:]
    values = np.arange(-10, 11, 1)

    # fixme
    model.train(False)

    for split in ['train', 'test']:
        inputs, targets = iter(loaders[split]).next()
        inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

        # forward
        outputs, base = model.forward(inputs, returns = ['z'])

        for dim in dimensions:
            code = to_np(base).copy()

            samples = []
            for val in values:
                # forward
                code[:, dim] = val
                sample = model.forward(inputs, z = to_var(code, volatile = True))

                # visualize
                sample = visualize(inputs, sample)
                samples.append(sample)

            # save
            for k in range(args.batch):
                images = [sample[k] for sample in samples]
                image_path = os.path.join(images_path, '{0}-{1}-{2}.gif'.format(split, k, dim))
                save_images(images, image_path, duration = .1, channel_first = True)

    # visualization
    with open(os.path.join(test_path, 'index.html'), 'w') as fp:
        for dim in dimensions:
            print('<h3>dimension [{0}]</h3>'.format(dim), file = fp)

            # table
            print('<table border="1" style="table-layout: fixed;">', file = fp)
            for split in ['train', 'test']:
                print('<tr>', file = fp)
                for k in range(args.batch):
                    image_path = os.path.join('images', '{0}-{1}-{2}.gif'.format(split, k, dim))
                    print('<td halign="center" style="word-wrap: break-word;" valign="top">', file = fp)
                    print('<img src="{0}" style="width:128px;">'.format(image_path), file = fp)
                    print('</td>', file = fp)
                print('</tr>', file = fp)
            print('</table>', file = fp)
