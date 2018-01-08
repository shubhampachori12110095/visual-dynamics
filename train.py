from __future__ import print_function

import argparse
import os

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import MotionDataset
from misc import visualize
from networks import VDNet
from utils.torch import Logger, kld_loss, load_snapshot, save_snapshot, to_np, to_var

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
    parser.add_argument('--batch', default = 32, type = int)

    # adaptive beta
    parser.add_argument('--beta', default = 0.00001, type = float)
    parser.add_argument('--max_beta', default = np.inf, type = float)
    parser.add_argument('--target_loss', default = 10, type = float)

    # training
    parser.add_argument('--epochs', default = 256, type = int)
    parser.add_argument('--snapshot', default = 8, type = int)
    parser.add_argument('--learning_rate', default = 0.001, type = float)

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

    # experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = False)

    # logger
    logger = Logger(exp_path)

    # load snapshot
    if args.resume is not None:
        epoch, args.beta = load_snapshot(args.resume, model = model, optimizer = optimizer, returns = ['epoch', 'beta'])
        print('==> snapshot "{0}" loaded (with beta = {1})'.format(args.resume, args.beta))
    else:
        epoch = 0

    # iterations
    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])
        print('==> epoch {0} (starting from step {1})'.format(epoch + 1, step + 1))

        # training
        model.train()
        for inputs, targets in tqdm(loaders['train'], desc = 'train'):
            inputs, targets = to_var(inputs), to_var(targets)

            # forward
            optimizer.zero_grad()
            outputs, (mean, log_var) = model.forward(inputs, returns = ['mean', 'log_var'])

            # reconstruction & kl divergence loss
            loss_r = mse_loss(outputs, targets)
            loss_kl = kld_loss(mean, log_var)

            # overall loss
            loss = loss_r + args.beta * loss_kl

            # logger
            logger.scalar_summary('train-loss', loss.data[0], step)
            logger.scalar_summary('train-loss-r', loss_r.data[0], step)
            logger.scalar_summary('train-loss-kl', loss_kl.data[0], step)
            step += targets.size(0)

            # backward
            loss.backward()
            optimizer.step()

        # testing
        model.train(False)

        loss_r, loss_kl = 0, 0
        for inputs, targets in tqdm(loaders['test'], desc = 'test'):
            inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

            # forward
            outputs, (mean, log_var) = model.forward(inputs, returns = ['mean', 'log_var'])

            # reconstruction & kl divergence loss
            loss_r += mse_loss(outputs, targets) * targets.size(0) / len(data['test'])
            loss_kl += kld_loss(mean, log_var) * targets.size(0) / len(data['test'])

        logger.scalar_summary('test-loss-r', loss_r.data[0], step)
        logger.scalar_summary('test-loss-kl', loss_kl.data[0], step)

        # beta
        if args.target_loss is not None and loss_r.data[0] < args.target_loss:
            if loss_kl.data[0] * args.beta < loss_r.data[0] and args.beta < args.max_beta:
                args.beta = min(args.beta * 2, args.max_beta)
                print('==> adjusted beta to {0}'.format(args.beta))

        # means & log_vars
        num_dists = 1024

        means, log_vars = [], []
        for inputs, targets in loaders['train']:
            inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

            # forward
            outputs, (mean, log_var) = model.forward(inputs, returns = ['mean', 'log_var'])

            means.extend(to_np(mean))
            log_vars.extend(to_np(log_var))

            if len(means) >= num_dists and len(log_vars) >= num_dists:
                break

        means = np.array(means[:num_dists])
        log_vars = np.array(log_vars[:num_dists])

        # visualization
        num_samples = 4

        for split in ['train', 'test']:
            inputs, targets = iter(loaders[split]).next()
            inputs, targets = to_var(inputs, volatile = True), to_var(targets, volatile = True)

            # forward (recontruction)
            outputs = model.forward(inputs)

            # forward (sampling)
            samples = []
            for k in range(num_samples):
                indices = np.random.choice(num_dists, args.batch)
                sample = model.forward(inputs, mean = to_var(means[indices]), log_var = to_var(log_vars[indices]))
                samples.append(sample)

            # visualize
            outputs = visualize(inputs, outputs)
            targets = visualize(inputs, targets)
            samples = [visualize(inputs, sample) for sample in samples]
            inputs = visualize(inputs)

            # logger
            logger.image_summary('{0}-inputs'.format(split), inputs, step)
            logger.image_summary('{0}-outputs'.format(split), zip(inputs, outputs), step)
            logger.image_summary('{0}-targets'.format(split), zip(inputs, targets), step)

            for k, sample in enumerate(samples):
                logger.image_summary('{0}-samples-{1}'.format(split, k + 1), zip(inputs, sample), step)

        # snapshot
        save_snapshot(os.path.join(exp_path, 'latest.pth'),
                      model = model, optimizer = optimizer, epoch = epoch + 1,
                      beta = args.beta, means = means, log_vars = log_vars)

        if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
            save_snapshot(os.path.join(exp_path, 'epoch-{0}.pth'.format(epoch + 1)),
                          model = model, optimizer = optimizer, epoch = epoch + 1,
                          beta = args.beta, means = means, log_vars = log_vars)
