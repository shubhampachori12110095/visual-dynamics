from __future__ import print_function

import torch.nn as nn


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        pass

    def forward(self, inputs):
        mu, sigma = None, None
        return mu, sigma


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
    def __init__(self):
        super(VDNet, self).__init__()

        self.image_encoder = ImageEncoder()
        self.motion_encoder = MotionEncoder()
        self.kernel_decoder = KernelDecoder()
        self.motion_decoder = MotionDecoder()
        self.l = nn.Linear(1, 1)

    def forward(self, inputs):
        # f = self.image_encoder.forward(inputs)
        # mu, sigma = self.motion_encoder.forward(inputs)
        #
        # z = None
        #
        # k = self.kernel_decoder.forward(z)
        #
        # m = F.conv(f, k)
        #
        # outputs = self.motion_decoder.forward(m)
        # return outputs
        pass
