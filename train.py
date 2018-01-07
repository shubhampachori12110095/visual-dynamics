from __future__ import print_function

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from data import MotionDataset
from misc import visualize
from networks import VDNet
from utils.torch import Logger, load_snapshot, save_snapshot, to_np, to_var

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

    # optimization
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--beta', default = 0.00001, type = float)
    parser.add_argument('--max_beta', default = np.inf, type = float)
    parser.add_argument('--target_loss', default = 10., type = float)

    # training
    parser.add_argument('--epochs', default = 1024, type = int)
    parser.add_argument('--snapshot', default = 1, type = int)

    # testing
    parser.add_argument('--size', default = 1024, type = int)
    parser.add_argument('--samples', default = 4, type = int)

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

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    print('==> optimizer loaded')

    # experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = False)

    # logger
    logger = Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))

    # load snapshot
    if args.resume is not None:
        snapshot = load_snapshot(args.resume, model = model, optimizer = optimizer)
        epoch = snapshot['epoch']
        print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
    else:
        epoch = 0

    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])

        # # training
        # model.train()
        # for inputs, targets in tqdm(loaders['train'], desc = 'epoch {0} train'.format(epoch + 1)):
        #     inputs, targets = to_var(inputs), to_var(targets)
        #
        #     # forward
        #     optimizer.zero_grad()
        #     outputs, (mean, log_var) = model.forward(inputs, params = ['mean', 'log_var'])
        #
        #     # reconstruction & kl divergence loss
        #     loss_r = mse_loss(outputs, targets)
        #     loss_kl = kld_loss(mean, log_var)
        #
        #     # overall loss
        #     loss = loss_r + args.beta * loss_kl
        #
        #     # scalar summary
        #     logger.scalar_summary('train_loss', loss.data[0], step)
        #     logger.scalar_summary('train_loss_r', loss_r.data[0], step)
        #     logger.scalar_summary('train_loss_kl', loss_kl.data[0], step)
        #     step += targets.size(0)
        #
        #     # backward
        #     loss.backward()
        #     optimizer.step()
        #
        # # testing
        model.train(False)
        #
        # loss_r, loss_kl = 0, 0
        # for inputs, targets in tqdm(loaders['test'], desc = 'epoch {0} test'.format(epoch + 1)):
        #     inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)
        #
        #     # forward
        #     outputs, (mean, log_var) = model.forward(inputs, params = ['mean', 'log_var'])
        #
        #     # reconstruction & kl divergence loss
        #     loss_r += mse_loss(outputs, targets) * targets.size(0) / len(data['test'])
        #     loss_kl += kld_loss(mean, log_var) * targets.size(0) / len(data['test'])
        #
        # # scalar summary
        # logger.scalar_summary('test_loss_r', loss_r.data[0], step)
        # logger.scalar_summary('test_loss_kl', loss_kl.data[0], step)
        #
        # # adjust beta
        # if args.target_loss is not None and loss_r.data[0] < args.target_loss:
        #     if loss_kl.data[0] * args.beta < loss_r.data[0] and args.weight_kl < args.max_beta:
        #         args.beta = min(args.beta * 2, args.max_beta)
        #         print('==> adjusted beta to {0}'.format(args.beta))

        # means & log_vars
        means, log_vars = [], []
        for inputs, targets in loaders['train']:
            inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

            # forward
            outputs, (mean, log_var) = model.forward(inputs, params = ['mean', 'log_var'])

            means.extend(to_np(mean).tolist())
            log_vars.extend(to_np(log_var).tolist())

            if len(means) >= args.size and len(log_vars) >= args.size:
                break

        means = np.array(means[:args.size])
        log_vars = np.array(log_vars[:args.size])

        # visualization
        for split in ['train', 'test']:
            inputs, targets = iter(loaders[split]).next()
            inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

            # forward (recontruction)
            outputs = model.forward(inputs)

            # forward (sampling)
            samples = []
            for k in range(args.samples):
                indices = np.random.choice(args.size, args.batch)
                sample = model.forward(inputs, mean = to_var(means[indices]), log_var = to_var(log_vars[indices]))
                samples.append(sample)

            # visualize
            outputs = visualize(inputs, outputs)
            targets = visualize(inputs, targets)
            samples = [visualize(inputs, sample) for sample in samples]
            inputs = visualize(inputs)

            # image summary
            logger.image_summary('{0}-inputs'.format(split), inputs, step)
            logger.image_summary('{0}-outputs'.format(split), zip(inputs, outputs), step)
            logger.image_summary('{0}-targets'.format(split), zip(inputs, targets), step)

            for k, sample in enumerate(samples):
                logger.image_summary('{0}-samples-{1}'.format(split, k + 1), zip(inputs, sample), step)

        # snapshot
        if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
            save_snapshot(os.path.join(exp_path), epoch + 1, model = model, optimizer = optimizer,
                          beta = args.beta, z = (means, log_vars))
            print('==> saved snapshot to "{0}"'.format(exp_path))
