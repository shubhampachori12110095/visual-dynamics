import os

from torch.utils.data import Dataset

from utils.image import load_image


class MotionDataset(Dataset):
    def __init__(self, data_path, split, scales = [.25, .5, 1, 2]):
        # settings
        self.path = data_path
        self.scales = scales

        # dataset list
        self.data = open(os.path.join(data_path, '{0}.txt'.format(split))).read().splitlines()

    def __getitem__(self, index):
        # motion inputs
        m1 = load_image(os.path.join(self.path, '{0}_im1.png'.format(index)), channel_first = True)
        m2 = load_image(os.path.join(self.path, '{0}_im2.png'.format(index)), channel_first = True)
        m_inputs = (m1, m2)

        # image inputs
        i_inputs = []
        for scale in self.scales:
            size = int(m1.shape[-1] * scale)
            i = load_image(os.path.join(self.path, '{0}_im1.png'.format(index)), size = size, channel_first = True)
            i_inputs.append(i)

        # inputs & targets
        inputs = (i_inputs, m_inputs)
        targets = m2 - m1
        return inputs, targets

    def __len__(self):
        return len(self.data)
