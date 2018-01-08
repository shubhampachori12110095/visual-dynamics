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
    parser.add_argument('--data_path', default = None)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 16, type = int)

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
    model.train(False)

    # test path
    test_path = os.path.join('exp', args.exp, 'test')
    utils.shell.mkdir(test_path, clean = True)

    # images path
    images_path = os.path.join(test_path, 'images')
    utils.shell.mkdir(images_path, clean = True)

    # load snapshot
    means, log_vars = load_snapshot(args.resume, model = model, returns = ['means', 'log_vars'])

    # statistics
    num_dists, num_dims = means.shape

    x, ym, yv = [], [], []
    for k in range(num_dims):
        x.extend([k] * num_dists)
        ym.extend(means[:, k])
        yv.extend(log_vars[:, k])

    plt.figure()
    plt.plot(x, ym, color = 'b')
    plt.xlabel('dimension')
    plt.ylabel('mean')
    plt.savefig(os.path.join(images_path, 'means.png'), bbox_inches = 'tight')

    plt.figure()
    plt.plot(x, yv, color = 'b')
    plt.xlabel('dimension')
    plt.ylabel('log(var)')
    plt.savefig(os.path.join(images_path, 'vars.png'), bbox_inches = 'tight')

    # dimensions
    bound, step = 8., .5
    values = np.arange(-bound, bound + step, step)

    threshold = .1
    deviation = np.std(means, axis = 0)
    indices = np.argsort(-deviation)

    dimensions = indices[np.where(deviation[indices] > threshold)[0]]
    print('==> dominated dimensions = {0}'.format(dimensions.tolist()))

    for split in ['train', 'test']:
        inputs, targets = iter(loaders[split]).next()
        inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

        # base
        outputs, base = model.forward(inputs, returns = ['z'])

        for dim in dimensions:
            # code
            code = to_np(base).copy()

            # forward
            samples = []
            for val in values:
                code[:, dim] = val
                sample = model.forward(inputs, z = to_var(code, volatile = True))
                samples.append(visualize(inputs, sample))

            # save images
            for k in range(args.batch):
                images = [sample[k] for sample in samples]
                image_path = os.path.join(images_path, '{0}-{1}-{2}.gif'.format(split, k, dim))
                save_images(images, image_path, duration = .1, channel_first = True)

    # visualization
    with open(os.path.join(test_path, 'index.html'), 'w') as fp:
        # statistics
        print('<h3>statistics</h3>', file = fp)
        print('<img src="{0}">'.format(os.path.join('images', 'means.png')), file = fp)
        print('<img src="{0}">'.format(os.path.join('images', 'vars.png')), file = fp)

        # dimensions
        for dim in dimensions:
            print('<h3>dimension [{0}]</h3>'.format(dim), file = fp)
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
