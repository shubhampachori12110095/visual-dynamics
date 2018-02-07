from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch import ConvPool2D, conv_cross2d, gaussian_sampler, weights_init


class ImageEncoder(nn.Module):
    def __init__(self, num_scales, channels, kernel_sizes, sampling_sizes):
        super(ImageEncoder, self).__init__()
        self.num_scales = num_scales
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.sampling_sizes = sampling_sizes

        # modules
        self.encoders = []
        for k in range(num_scales):
            self.encoders.append(ConvPool2D(
                channels = self.channels,
                kernel_sizes = self.kernel_sizes,
                last_nonlinear = True,
                sampling_type = 'SUB-MAXPOOL',
                sampling_sizes = self.sampling_sizes
            ))
            self.add_module('encoder-{0}'.format(k + 1), self.encoders[-1])
        self.apply(weights_init)

    def forward(self, inputs):
        outputs = [encoder.forward(input) for encoder, input in zip(self.encoders, inputs)]
        return outputs


class MotionEncoder(nn.Module):
    def __init__(self, channels, kernel_sizes, sampling_sizes):
        super(MotionEncoder, self).__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.sampling_sizes = sampling_sizes

        # modules
        self.encoder = ConvPool2D(
            channels = self.channels,
            kernel_sizes = self.kernel_sizes,
            last_nonlinear = False,
            sampling_type = 'SUB-MAXPOOL',
            sampling_sizes = self.sampling_sizes
        )
        self.apply(weights_init)

    def forward(self, inputs):
        inputs = torch.cat(inputs, 1)
        outputs = self.encoder.forward(inputs)
        outputs = outputs.view(inputs.size(0), -1)

        mean, log_var = torch.split(outputs, outputs.size(1) // 2, 1)
        return mean, log_var


class KernelDecoder(nn.Module):
    def __init__(self, num_scales, in_channels, out_channels, kernel_size, num_groups, num_layers, kernel_sizes):
        super(KernelDecoder, self).__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes

        # channels
        self.num_channels = self.num_scales * self.out_channels * (self.in_channels // self.num_groups)

        # modules
        self.decoder = nn.Sequential(
            ConvPool2D(
                channels = [self.num_channels] * 2,
                sampling_type = 'UP-DECONV',
                sampling_sizes = self.kernel_size
            ),
            ConvPool2D(
                channels = [self.num_channels] * self.num_layers,
                kernel_sizes = self.kernel_sizes
            ),
            nn.BatchNorm2d(self.num_channels)
        )
        self.apply(weights_init)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_channels, 1, 1)
        outputs = self.decoder.forward(inputs)
        outputs = outputs.view(-1, self.num_scales, self.out_channels, self.in_channels // self.num_groups,
                               self.kernel_size, self.kernel_size)
        return outputs


class MotionDecoder(nn.Module):
    def __init__(self, scales, channels, kernel_sizes):
        super(MotionDecoder, self).__init__()
        self.scales = scales
        self.channels = channels
        self.kernel_sizes = kernel_sizes

        # modules
        self.decoder = ConvPool2D(
            channels = self.channels,
            kernel_sizes = self.kernel_sizes
        )
        self.apply(weights_init)

    def forward(self, inputs):
        for k, input in enumerate(inputs):
            scale_factor = int(self.scales[-1] / self.scales[k])
            if scale_factor != 1:
                inputs[k] = F.upsample(input, scale_factor = scale_factor, mode = 'nearest')

        inputs = torch.cat(inputs, 1)
        outputs = self.decoder.forward(inputs)
        return outputs


class VDNet(nn.Module):
    def __init__(self, scales = [.25, .5, 1, 2]):
        super(VDNet, self).__init__()
        self.scales = scales

        # modules
        self.image_encoder = ImageEncoder(
            num_scales = len(scales),
            channels = [3, 64, 64, 64, 32],
            sampling_sizes = [2, 1, 2, 1],
            kernel_sizes = 5
        )
        self.motion_encoder = MotionEncoder(
            channels = [6, 96, 96, 128, 128, 256, 256],
            sampling_sizes = [4, 2, 2, 2, 2, 2],
            kernel_sizes = 5
        )
        self.kernel_decoder = KernelDecoder(
            num_scales = len(scales),
            in_channels = 32,
            out_channels = 32,
            num_groups = 32,
            kernel_size = 5,
            num_layers = 3,
            kernel_sizes = 5
        )
        self.motion_decoder = MotionDecoder(
            scales = scales,
            channels = [len(scales) * 32, 128, 128, 3],
            kernel_sizes = [9, 1, 1]
        )
        self.apply(weights_init)

    def forward(self, inputs, mean = None, log_var = None, z = None, returns = None):
        if isinstance(inputs, list) and len(inputs) == 2:
            i_inputs, m_inputs = inputs
        else:
            i_inputs, m_inputs = inputs, None

        # image encoder
        features = self.image_encoder.forward(i_inputs)

        # motion encoder
        if mean is None and log_var is None and z is None:
            mean, log_var = self.motion_encoder.forward(m_inputs)

        # gaussian sampler
        if z is None:
            z = gaussian_sampler(mean, log_var)

        # kernel decoder
        kernels = self.kernel_decoder.forward(z)

        # cross convolution
        for i, feature in enumerate(features):
            kernel = kernels[:, i, ...].contiguous()
            padding = (kernel.size(-1) - 1) // 2
            num_groups = feature.size(1) // kernel.size(2)
            features[i] = conv_cross2d(feature, kernel, padding = padding, groups = num_groups)

        # motion decoder
        outputs = self.motion_decoder.forward(features)

        # returns
        if returns is not None:
            for i, k in enumerate(returns):
                returns[i] = locals()[k]
            return outputs, returns[0] if len(returns) == 1 else returns

        return outputs
