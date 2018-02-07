from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils
from data import MotionDataset
from misc import visualize
from networks import VDNet
from utils.image import resize_image, save_images
from utils.torch import load_snapshot, to_np, to_var


def analyze_fmaps(size = 256):
    fmaps_path = os.path.join('exp', args.exp, 'fmaps')
    images_path = os.path.join(fmaps_path, 'images')
    utils.shell.mkdir(images_path, clean = True)

    # feature maps
    for split in ['train', 'test']:
        inputs, targets = iter(loaders[split]).next()
        inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

        outputs, features = model.forward(inputs, returns = ['features'])
        num_scales, num_channels = len(features), features[0].size(1)

        for s in trange(num_scales):
            input, feature = inputs[0][s], features[s]

            for b in trange(args.batch, leave = False):
                image = resize_image(to_np(input[b]), size = size, channel_first = True)

                for c in trange(num_channels, leave = False):
                    fmap = resize_image(to_np(feature[b, c]), size = size, channel_first = True)

                    # normalize
                    if np.min(fmap) < np.max(fmap):
                        fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap))

                    # save images
                    image_path = os.path.join(images_path, '{0}-{1}-{2}-{3}.gif'.format(split, s, c, b))
                    save_images([image, fmap], image_path, channel_first = True)

    # visualization
    with open(os.path.join(fmaps_path, 'index.html'), 'w') as fp:
        for s in range(num_scales):
            for c in range(num_channels):
                print('<h3>scale [{0}] - channel [{1}]</h3>'.format(s + 1, c + 1), file = fp)
                print('<table border="1" style="table-layout: fixed;">', file = fp)
                for split in ['train', 'test']:
                    print('<tr>', file = fp)
                    for b in range(args.batch):
                        image_path = os.path.join('images', '{0}-{1}-{2}-{3}.gif'.format(split, s, c, b))
                        print('<td halign="center" style="word-wrap: break-word;" valign="top">', file = fp)
                        print('<img src="{0}" style="width:128px;">'.format(image_path), file = fp)
                        print('</td>', file = fp)
                    print('</tr>', file = fp)
                print('</table>', file = fp)


def analyze_reprs(max_dims = 16, threshold = .5, bound = 8., step = .2):
    reprs_path = os.path.join('exp', args.exp, 'reprs')
    images_path = os.path.join(reprs_path, 'images')
    utils.shell.mkdir(images_path, clean = True)

    # statistics
    x, ym, yv = [], [], []
    for k in range(means.shape[1]):
        x.extend([k, k])
        ym.extend([np.min(means[:, k]), np.max(means[:, k])])
        yv.extend([np.min(log_vars[:, k]), np.max(log_vars[:, k])])

    plt.figure()
    plt.bar(x, ym, .5, color = 'b')
    plt.xlabel('dimension')
    plt.ylabel('mean')
    plt.savefig(os.path.join(images_path, 'means.png'), bbox_inches = 'tight')

    plt.figure()
    plt.bar(x, yv, .5, color = 'b')
    plt.xlabel('dimension')
    plt.ylabel('log(var)')
    plt.savefig(os.path.join(images_path, 'vars.png'), bbox_inches = 'tight')

    # dimensions
    values = np.arange(-bound, bound + step, step)

    magnitudes = np.max(np.abs(means), axis = 0)
    indices = np.argsort(-magnitudes)

    dimensions = [k for k in indices if magnitudes[k] > threshold][:max_dims]
    print('==> dominated dimensions = {0}'.format(dimensions))

    for split in ['train', 'test']:
        inputs, targets = iter(loaders[split]).next()
        inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

        outputs, z = model.forward(inputs, returns = ['z'])

        for dim in tqdm(dimensions):
            repr = to_np(z).copy()

            # forward
            samples = []
            for val in tqdm(values, leave = False):
                repr[:, dim] = val
                sample = model.forward(inputs, z = to_var(repr, volatile = True))
                samples.append(visualize(inputs, sample))

            # save images
            for k in range(args.batch):
                images = [sample[k] for sample in samples]
                image_path = os.path.join(images_path, '{0}-{1}-{2}.gif'.format(split, k, dim))
                save_images(images, image_path, duration = .1, channel_first = True)

    # visualization
    with open(os.path.join(reprs_path, 'index.html'), 'w') as fp:
        print('<h3>statistics</h3>', file = fp)
        print('<img src="{0}">'.format(os.path.join('images', 'means.png')), file = fp)
        print('<img src="{0}">'.format(os.path.join('images', 'vars.png')), file = fp)

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
    model.train(False)

    # load snapshot
    if args.resume is None:
        args.resume = os.path.join('exp', args.exp, 'latest.pth')

    means, log_vars = load_snapshot(args.resume, model = model, returns = ['means', 'log_vars'])

    # analyze feature maps
    print('==> analyzing feature maps')
    analyze_fmaps()

    # analyze representations
    print('==> analyzing representations')
    analyze_reprs()
