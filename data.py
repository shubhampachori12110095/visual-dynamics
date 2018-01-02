import os

from torch.utils.data import Dataset

from utils.image import load_image


class MotionDataset(Dataset):
    def __init__(self, data_path, split, mode = 'DIFF'):
        # settings
        self.path = data_path
        self.mode = mode.upper()

        # data list
        self.data = open(os.path.join(data_path, '{0}.txt'.format(split))).read().splitlines()

    def __getitem__(self, index):
        if self.mode == 'DIFF':
            im1 = load_image(os.path.join(self.path, '{0}_im1.png'.format(index)), channel_first = True)
            im2 = load_image(os.path.join(self.path, '{0}_im2.png'.format(index)), channel_first = True)

            inputs, targets = (im1, im2), (im2 - im1)
        else:
            raise NotImplementedError('unsupported dataset mode "{0}"'.format(self.mode))

        return inputs, targets

    def __len__(self):
        return len(self.data)
