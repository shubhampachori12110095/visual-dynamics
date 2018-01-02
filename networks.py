from __future__ import print_function

import torch
import torch.nn as nn

from utils.torch import ConvPool2D, GaussianSampler, weights_init


class ImageEncoder(nn.Module):
    def __init__(self, scales, channels, kernal_sizes, sampling_sizes):
        super(ImageEncoder, self).__init__()

        # settings
        self.scales = scales

        # encoders
        self.encoders = []
        for k, scale in enumerate(scales):
            self.encoders.append(ConvPool2D(channels = channels, kernel_sizes = kernal_sizes,
                                            batch_norm = True, nonlinear_type = 'RELU', last_nonlinear = True,
                                            sampling_type = 'SUB-MAXPOOL', sampling_sizes = sampling_sizes))
            self.add_module('encoder-{0}'.format(k + 1), self.encoders[-1])

    def forward(self, inputs):
        outputs = []
        for k, input in enumerate(inputs):
            output = self.encoders[k].forward(input)
            outputs.append(output)
        return outputs


class MotionEncoder(nn.Module):
    def __init__(self, channels, kernal_sizes, sampling_sizes):
        super(MotionEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            ConvPool2D(channels = channels, kernel_sizes = kernal_sizes,
                       batch_norm = True, nonlinear_type = 'RELU', last_nonlinear = True,
                       sampling_type = 'SUB-MAXPOOL', sampling_sizes = sampling_sizes),
            nn.Conv2d(channels[-1], channels[-1], 4, 1, 0)
        )
        self.encoder.apply(weights_init)

        # sampler
        self.sampler = GaussianSampler()

    def forward(self, inputs):
        inputs = torch.cat(inputs, 1)

        # input => output
        outputs = self.encoder.forward(inputs)
        outputs = outputs.view(inputs.size(0), -1)

        # output => mean, logvar
        params = torch.split(outputs, outputs.size(1) // 2, 1)

        # mean, logvar => z
        z = self.sampler.forward(params)
        return z, params


class KernelDecoder(nn.Module):
    def __init__(self, num_scales, in_channels, out_channels, kernel_size):
        super(KernelDecoder, self).__init__()
        pass

    def forward(self, inputs):
        pass


class MotionDecoder(nn.Module):
    def __init__(self):
        super(MotionDecoder, self).__init__()
        pass

    def forward(self, inputs):
        pass


class VDNet(nn.Module):
    def __init__(self, scales = [.25, .5, 1, 2]):
        super(VDNet, self).__init__()

        # settings
        self.scales = scales

        # image encoder
        self.image_encoder = ImageEncoder(scales = scales,
                                          channels = [3, 64, 64, 64, 32],
                                          kernal_sizes = 5,
                                          sampling_sizes = [2, 1, 2, 1])

        # motion encoder
        self.motion_encoder = MotionEncoder(channels = [6, 96, 96, 128, 128, 256, 256],
                                            kernal_sizes = 5,
                                            sampling_sizes = [4, 1, 2, 1, 2, 1])

        # self.kernel_decoder = KernelDecoder()
        # self.motion_decoder = MotionDecoder()

    def forward(self, inputs, sampling_type = 'NONE'):
        # sanity check

        if sampling_type == 'NONE':
            i_inputs, m_inputs = inputs
        else:
            i_inputs = inputs

        f = self.image_encoder.forward(i_inputs)
        print(f)

        if sampling_type == 'NONE':
            z, params = self.motion_encoder.forward(m_inputs)
        else:
            assert False

        # k = self.kernel_decoder.forward(z)
        #
        #
        # m = F.conv(f, k)
        #
        # outputs = self.motion_decoder.forward(m)
        # return outputs
        pass
