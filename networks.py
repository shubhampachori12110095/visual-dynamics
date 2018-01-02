from __future__ import print_function

import torch
import torch.nn as nn

from utils.torch import weights_init


class ConvPool2D(nn.Module):
    def __init__(self, channels, kernel_sizes = 3, paddings = None,
                 batch_norm = True, nonlinear_type = 'LEAKY-RELU', last_nonlinear = True,
                 sampling_type = 'NONE', sampling_sizes = 1):
        super(ConvPool2D, self).__init__()

        num_layers = len(channels) - 1

        # int => list
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * num_layers
        if not isinstance(paddings, list):
            paddings = [paddings] * num_layers
        if not isinstance(sampling_sizes, list):
            sampling_sizes = [sampling_sizes] * num_layers

        modules = []
        for k in range(num_layers):
            in_channels, out_channels = channels[k], channels[k + 1]
            kernel_size, padding, sampling_size = kernel_sizes[k], paddings[k], sampling_sizes[k]

            # default padding size
            if padding is None:
                padding = (kernel_size - 1) // 2

            if sampling_type == 'UP-DECONV' and sampling_size != 1:
                modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size + 1, sampling_size, padding))
            if sampling_type == 'SUB-CONV' and sampling_size != 1:
                modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, sampling_size, padding))
            else:
                modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding))

            # batchnorm & nonlinear
            if last_nonlinear or k + 1 != num_layers:
                if batch_norm:
                    modules.append(nn.BatchNorm2d(out_channels))

                if nonlinear_type == 'RELU':
                    modules.append(nn.ReLU(True))
                elif nonlinear_type == 'LEAKY-RELU':
                    modules.append(nn.LeakyReLU(0.2, True))

            if sampling_type == 'SUB_AVGPOOL' and sampling_size != 1:
                modules.append(nn.AvgPool2d(sampling_size))
            elif sampling_type == 'SUB_MAXPOOL' and sampling_size != 1:
                modules.append(nn.MaxPool2d(sampling_size))
            elif sampling_type == 'UP_NEAREST' and sampling_size != 1:
                modules.append(nn.UpsamplingNearest2d(sampling_size))
            elif sampling_type == 'UP_BILINEAR' and sampling_size != 1:
                modules.append(nn.UpsamplingBilinear2d(sampling_size))

        self.network = nn.Sequential(*modules)
        self.network.apply(weights_init)

    def forward(self, inputs):
        return self.network.forward(inputs)


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()

        self.network = nn.Sequential(
            ConvPool2D(channels = [6, 128, 128, 512], kernel_sizes = 5,
                       sampling_type = 'SUB-MAXPOOL', sampling_sizes = 2),
            nn.Conv2d(512, 512, 4, 1, 0)
        )

    def forward(self, inputs):
        inputs = torch.cat(inputs, 1)
        print(type(inputs))
        outputs = self.network.forward(inputs)
        print(outputs.size())

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
