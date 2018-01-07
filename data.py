import os

from torch.utils.data import Dataset

from utils.image import load_image, resize_image


class MotionDataset(Dataset):
    def __init__(self, data_path, split, input_size = 128, input_scales = [.25, .5, 1, 2], target_size = 64):
        # settings
        self.path = data_path
        self.input_size = input_size
        self.input_scales = input_scales
        self.target_size = target_size

        # dataset list
        self.data = open(os.path.join(data_path, '{0}.txt'.format(split))).read().splitlines()

    def __getitem__(self, index):
        # motion inputs
        m1 = load_image(os.path.join(self.path, '{0}_im1.png'.format(self.data[index])),
                        size = self.input_size, channel_first = True)
        m2 = load_image(os.path.join(self.path, '{0}_im2.png'.format(self.data[index])),
                        size = self.input_size, channel_first = True)
        m_inputs = (m1, m2)

        # image inputs
        i_inputs = [resize_image(m1, size = int(self.input_size * input_scale), channel_first = True)
                    for input_scale in self.input_scales]

        # inputs & targets
        inputs = (i_inputs, m_inputs)
        targets = resize_image(m2, size = self.target_size, channel_first = True) - \
                  resize_image(m1, size = self.target_size, channel_first = True)

        return inputs, targets * 128.

    def __len__(self):
        return len(self.data)
