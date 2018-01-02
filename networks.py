from __future__ import print_function

import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()

    def forward(self, inputs):
        inputs = torch.cat(inputs, 1)

        mean, logvar = None, None
        return mean, logvar


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
        self.motion_encoder = MotionEncoder()
        self.kernel_decoder = KernelDecoder()
        self.motion_decoder = MotionDecoder()
        self.l = nn.Linear(1, 1)

    def forward(self, inputs):
        if self.training:
            mean, logvar = self.motion_encoder.forward(inputs)
            z = None
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
