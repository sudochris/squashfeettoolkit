import numpy as np


def _normalize(image, source=None, target=None):
    """
    Normalizes an image from [0, max] to the given target range.
    If no target range is given, [0, 1] is used
    :param image:
    :param target:
    :return:
    """
    if target is None:
        target = [0, 1]
    if source is None:
        source = [0, max(1, np.max(image))]
    return np.interp(image, source, target)


def linear(accumulator):
    """
    Performs a linear interpolation from [0, max] to [0, 255]
    :param accumulator:
    :return:
    """
    return _normalize(accumulator.get(), target=[0, 255]).astype(np.uint8)


def logarithmic(accumulator):
    """
    Performs a logarithmic transformation by ln(1+x)
    Then normalizes from [0, max] to [0, 255]
    :param accumulator:
    :return:
    """
    heatmap = np.log1p(accumulator.get())
    heatmap = _normalize(heatmap, target=[0, 255])
    return heatmap.astype(np.uint8)
