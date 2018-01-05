from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch import ConvPool2D, conv_cross2d, gaussian_sampler, weights_init


class ImageEncoder(nn.Module):
    def __init__(self, scales, channels, kernal_sizes, batch_norm, nonlinear_type, sampling_type, sampling_sizes):
        super(ImageEncoder, self).__init__()

        # settings
        self.scales = scales

        # encoders
        self.encoders = []
        for k, scale in enumerate(scales):
            self.encoders.append(ConvPool2D(channels = channels, kernel_sizes = kernal_sizes, batch_norm = batch_norm,
                                            nonlinear_type = nonlinear_type, last_nonlinear = True,
                                            sampling_type = sampling_type, sampling_sizes = sampling_sizes))
            self.add_module('encoder-{0}'.format(k + 1), self.encoders[-1])

    def forward(self, inputs):
        outputs = []
        for k, input in enumerate(inputs):
            output = self.encoders[k].forward(input)
            outputs.append(output)
        return outputs


class MotionEncoder(nn.Module):
    def __init__(self, channels, kernal_sizes, batch_norm, nonlinear_type, sampling_type, sampling_sizes):
        super(MotionEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            ConvPool2D(channels = channels, kernel_sizes = kernal_sizes,
                       batch_norm = batch_norm, nonlinear_type = nonlinear_type, last_nonlinear = True,
                       sampling_type = sampling_type, sampling_sizes = sampling_sizes),
            nn.Conv2d(channels[-1], channels[-1], 4, 1, 0)
        )
        self.encoder.apply(weights_init)

    def forward(self, inputs):
        inputs = torch.cat(inputs, 1)

        # input => output
        outputs = self.encoder.forward(inputs)
        outputs = outputs.view(inputs.size(0), -1)

        # output => mean, log_var
        mean, log_var = torch.split(outputs, outputs.size(1) // 2, 1)
        return mean, log_var


class KernelDecoder(nn.Module):
    def __init__(self, num_scales, in_channels, out_channels, kernel_size, num_groups,
                 num_layers, kernel_sizes, batch_norm, nonlinear_type, sampling_type, sampling_sizes):
        super(KernelDecoder, self).__init__()

        # settings
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        # channels
        self.num_channels = num_scales * out_channels * (in_channels // num_groups)

        # decoder
        self.decoder = nn.Sequential(
            ConvPool2D(channels = [self.num_channels] * num_layers, kernel_sizes = kernel_sizes,
                       batch_norm = batch_norm, nonlinear_type = nonlinear_type, last_nonlinear = False,
                       sampling_type = sampling_type, sampling_sizes = sampling_sizes),
            nn.BatchNorm2d(self.num_channels)
        )
        self.decoder.apply(weights_init)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_channels, self.kernel_size, self.kernel_size)
        outputs = self.decoder.forward(inputs)
        outputs = outputs.view(-1, self.num_scales, self.out_channels, self.in_channels // self.num_groups,
                               self.kernel_size, self.kernel_size)
        return outputs


class MotionDecoder(nn.Module):
    def __init__(self, scales, channels, kernal_sizes, batch_norm, nonlinear_type, sampling_type, sampling_sizes):
        super(MotionDecoder, self).__init__()

        # settings
        self.scales = scales

        # decoder
        self.decoder = ConvPool2D(channels = channels, kernel_sizes = kernal_sizes,
                                  batch_norm = batch_norm, nonlinear_type = nonlinear_type, last_nonlinear = False,
                                  sampling_type = sampling_type, sampling_sizes = sampling_sizes)

    def forward(self, inputs):
        # upsampling
        for k, input in enumerate(inputs):
            scale_factor = int(self.scales[-1] / self.scales[k])
            if scale_factor != 1:
                inputs[k] = F.upsample(input, scale_factor = scale_factor, mode = 'nearest')

        # inputs & outputs
        inputs = torch.cat(inputs, 1)
        outputs = self.decoder.forward(inputs)
        return outputs


class VDNet(nn.Module):
    def __init__(self, scales = [.25, .5, 1, 2]):
        super(VDNet, self).__init__()

        # settings
        self.scales = scales

        # image encoder
        self.image_encoder = ImageEncoder(scales = scales, channels = [3, 64, 64, 64, 32], kernal_sizes = 5,
                                          batch_norm = True, nonlinear_type = 'RELU',
                                          sampling_type = 'SUB-MAXPOOL', sampling_sizes = [2, 1, 2, 1])

        # motion encoder
        self.motion_encoder = MotionEncoder(channels = [6, 96, 96, 128, 128, 256, 256], kernal_sizes = 5,
                                            batch_norm = True, nonlinear_type = 'RELU',
                                            sampling_type = 'SUB-MAXPOOL', sampling_sizes = [4, 1, 2, 1, 2, 1])

        # kernel decoder
        self.kernel_decoder = KernelDecoder(num_scales = len(scales), in_channels = 32, out_channels = 32,
                                            kernel_size = 5, num_groups = 32,
                                            num_layers = 3, kernel_sizes = 5,
                                            batch_norm = True, nonlinear_type = 'RELU',
                                            sampling_type = 'NONE', sampling_sizes = 1)

        # motion decoder
        self.motion_decoder = MotionDecoder(scales = scales, channels = [len(scales) * 32, 128, 128, 3],
                                            kernal_sizes = [9, 1, 1], batch_norm = True, nonlinear_type = 'RELU',
                                            sampling_type = 'NONE', sampling_sizes = 1)

    def forward(self, inputs, mean = None, log_var = None, params = None):
        # inputs
        if isinstance(inputs, list) and len(inputs) == 2:
            i_inputs, m_inputs = inputs
        else:
            i_inputs, m_inputs = inputs, None

        # image encoder
        features = self.image_encoder.forward(i_inputs)

        # motion encoder
        if mean is None and log_var is None:
            # fixme
            assert m_inputs is not None
            mean, log_var = self.motion_encoder.forward(m_inputs)

        # sampler
        z = gaussian_sampler(mean, log_var)

        # kernel decoder
        kernels = self.kernel_decoder.forward(z)

        # cross convolution
        for k, feature in enumerate(features):
            kernel = kernels[:, k, ...].contiguous()

            # params
            padding = (kernel.size(-1) - 1) // 2
            num_groups = feature.size(1) // kernel.size(2)

            # cross convolution
            features[k] = conv_cross2d(feature, kernel, padding = padding, groups = num_groups)

        # motion decoder
        outputs = self.motion_decoder.forward(features)

        # params
        if params is not None:
            values = []
            for p in params:
                if p in locals():
                    values.append(locals()[p])

            if len(values) > 0:
                return outputs, values

        return outputs
