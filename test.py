from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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


    def show_dist(data, path):
        num, dim = data.shape

        x, y = [], []
        for k in range(dim):
            x.extend([k] * num)
            y.extend(data[:, k])

        plt.figure()
        plt.plot(x, y)
        plt.savefig(path, bbox_inches = 'tight')


    show_dist(means, 'means.png')
    show_dist(log_vars, 'log_vars.png')

    std = np.std(means, axis = 0)

    print(sorted(std)[::-1][:10])

    # testing
    model.train(False)

    """
        local selectedDimensions = {778, 2958, 2963, 2971, 3149}
        local nsample = math.min(100, traindata_handler:getsize())
        local zRange = torch.linspace(-10, 10, 21)
        local totalImage = nsample * zRange:size()[1] * #selectedDimensions
        local finished = 0
        for zdimIndex = 1, #selectedDimensions do
            local zdim = selectedDimensions[zdimIndex]

            for i=1,nsample,opt.batchSize do
                local inputs,targets = traindata_handler:getdata(batchidx)

                -- Get z value
                local z = crossconvmulti_getz(model.seqmodel, inputs)

                -- Forward propagation
                for zRangeIndex = 1, zRange:size(1) do
                    z[{{},zdim}] = zRange[zRangeIndex]
                    local outputs = crossconvmulti_sampling(model.seqmodel, z)
                    local outim = crossconvmulti_out2im(inputs, outputs)
                    for j=1,batchlen do
                        local outfile = path.join(dumppath, string.format('%03d_%02d.png', i+j, zRangeIndex))
                        image.save(outfile, outim[j])
                        table.insert(filelist[j], outfile)
                    end
                end

                for j = 1, batchlen do
                    local gifFile = string.format('%03d.gif', i+j)
                    table.insert(giflist, gifFile)
                    genGIF(filelist[j], path.join(dumppath, gifFile), 10)
                    for zRangeIndex = 1, zRange:size(1) do
                        os.remove(filelist[j][zRangeIndex])
                    end
                end
            end
            tftools.createHTMLimages(dumppath, 'index.html', giflist, string.format('z=%d',zdim))
        end
    """
