from scipy.interpolate import interp2d
import numpy as np


def bicubic_interpolation(data, factor) -> np.ndarray:
    """
    Parameters
    ----------
    data
    factor
    """
    data = np.array(data).squeeze()

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    if data.ndim == 3:
        old_shape = data.shape[1:]
        new_shape = tuple(sh * factor for sh in old_shape)
        ynew, xnew = (np.arange(sh) for sh in new_shape)
        yold = (ynew[factor - 1::factor] + ynew[:-factor + 1:factor]) / 2
        xold = (xnew[factor - 1::factor] + xnew[:-factor + 1:factor]) / 2
        z = np.zeros((data.shape[0], *new_shape))
        for i, di in enumerate(data):
            f = interp2d(xold, yold, di, kind='cubic')
            z[i] = f(xnew, ynew)
    else:
        raise ValueError(f'Invalid data for interpolation. Got shape {data.shape}.')

    return z.squeeze()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count