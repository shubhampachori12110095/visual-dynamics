import os

from torch.utils.data import Dataset
import scipy.misc


class MotionDataset(Dataset):
    def __init__(self, data_path, split, mode = 'DIFF'):
        self.path = data_path
        self.mode = mode.upper()

        self.data = open(os.path.join(data_path, '{0}.txt'.format(split))).read().splitlines()

    def __getitem__(self, index):
        if self.mode == 'DIFF':
            im1 = scipy.misc.imread(os.path.join(self.path, '{0}_im1.png'.format(index)))
            im2 = scipy.misc.imread(os.path.join(self.path, '{0}_im2.png'.format(index)))

            inputs = (im1, im2)
            targets = im2 - im1

        return inputs, targets

    def __len__(self):
        return len(self.data)
