import numpy as np


def image_mse(img1, img2):
    if type(img1) is not np.ndarray:
        raise Exception("Please input a numpy nd array")
    if type(img2) is not np.ndarray:
        raise Exception("Please input a numpy nd array")
    assert img1.shape == img2.shape, "Blocks are not in same shape"
    squared_error = np.square(img1 - img2)
    mse = squared_error.mean()

    return mse