import os

from torch.utils.data import Dataset

from utils.image import load_image, resize_image


class MotionDataset(Dataset):
    def __init__(self, data_path, split, input_size = 128, input_scales = [.25, .5, 1, 2], target_size = 64):
        self.data_path = data_path
        self.split = split
        self.input_size = input_size
        self.input_scales = input_scales
        self.target_size = target_size

        self.data = open(os.path.join(self.data_path, '{0}.txt'.format(self.split))).read().splitlines()

    def __getitem__(self, index):
        m_inputs, i_inputs = [], []

        for k in range(2):
            m_inputs.append(load_image(
                os.path.join(self.data_path, '{0}_im{1}.png'.format(self.data[index], k + 1)),
                size = self.input_size,
                channel_first = True
            ))

        for input_scale in self.input_scales:
            i_inputs.append(resize_image(m_inputs[0], size = int(self.input_size * input_scale), channel_first = True))

        inputs = (i_inputs, m_inputs)
        targets = resize_image(m_inputs[1], size = self.target_size, channel_first = True) - \
                  resize_image(m_inputs[0], size = self.target_size, channel_first = True)

        return inputs, targets * 128.

    def __len__(self):
        return len(self.data)
