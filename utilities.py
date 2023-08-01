import json
import numpy as np


def gaussian_filter(N=18, sigma=0.5):
    if not isinstance(sigma, (list, tuple, np.ndarray)) and not isinstance(N, (list, tuple, np.ndarray)):
        n = (N-1)/2.0
        y, x = np.ogrid[-n:n+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2*sigma**2))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    elif not isinstance(N, (list, tuple, np.ndarray)):
        return [gaussian_filter(N, s) for s in sigma]
    elif not isinstance(sigma, (list, tuple, np.ndarray)):
        return [gaussian_filter(n, sigma) for n in N]
    else:
        return [gaussian_filter(n, s) for n, s in zip(N,sigma)]


def gaussian_filters(N=7, sigmas=[0.42, 0.35, 0.35, 0.36, 0.2, 0.24]):
    return np.stack([gaussian_filter(N, sigma) for sigma in sigmas])


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
