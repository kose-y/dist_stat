import numpy as np
import torch
def breslow_ind(arr):
    arr_rev = np.flip(arr, 0)
    _, uind, uinv = np.unique(arr_rev, return_index=True, return_inverse=True)
    r = arr_rev.shape[0] - uind[uinv] - 1
    r = np.flip(r, 0).copy()
    return r
