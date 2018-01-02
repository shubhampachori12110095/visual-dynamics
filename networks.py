from __future__ import print_function

import torch
import torch.nn as nn

from utils.torch import ConvPool2D, GaussianSampler, weights_init


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
    def __init__(self):
        super(KernelDecoder, self).__init__()
        pass

    def forward(self, inputs):
        pass


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
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
    def __init__(self, scales = [1]):
        super(VDNet, self).__init__()

        # todo: multiple scales
        self.scales = scales

        self.image_encoder = ImageEncoder()
        self.motion_encoder = MotionEncoder(channels = [6, 96, 96, 128, 128, 256, 256], kernal_sizes = 5,
                                            sampling_sizes = [4, 1, 2, 1, 2, 1])
        self.kernel_decoder = KernelDecoder()
        self.motion_decoder = MotionDecoder()
        self.l = nn.Linear(1, 1)

    def forward(self, inputs):
        if self.training:
            z, params = self.motion_encoder.forward(inputs)
            print(z.size())
        else:
            z = None

        # k = self.kernel_decoder.forward(z)

        f = self.image_encoder.forward(inputs[0] if self.training else inputs)

        # f = self.image_encoder.forward(inputs)
        #
        #
        #
        # m = F.conv(f, k)
        #
        # outputs = self.motion_decoder.forward(m)
        # return outputs
        pass
