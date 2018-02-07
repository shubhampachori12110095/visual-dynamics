import numpy as np

from utils.image import resize_image
from utils.torch import to_np


def visualize(inputs, outputs = None, size = 256):
    if isinstance(inputs, list) and len(inputs) == 2:
        inputs = inputs[0][-1]
    else:
        inputs = inputs[-1]

    if outputs is None:
        outputs = np.zeros(inputs.size())

    images = []
    for input, output in zip(inputs, outputs):
        input, output = to_np(input), to_np(output)
        input = resize_image(input, output.shape[-1], channel_first = True)

        image = input + output / 128.
        image = np.maximum(image, 0)
        image = np.minimum(image, 1)

        image = resize_image(image, size, channel_first = True)
        images.append(image)
    return images
