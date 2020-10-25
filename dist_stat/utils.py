import numpy as np
import torch
def coo_to_sparsetensor(spm, TType=torch.sparse.DoubleTensor):
    i = np.vstack([spm.row, spm.col])
    v = spm.data
    return TType(i, v, spm.shape)
def gidx_to_partition(gidx):
    assert([gidx[i+1]>=gidx[i] for i in range(len(gidx)-1)])
    m = int(max(gidx))
    return [0]+[np.where(gidx<=i)[0].max()+1 for i in range(m+1)]
def find_nearest(arr, val):
    idx = np.abs(arr-val).argmin()
    return arr[idx]
def impute_na(arr):
    row_mean = np.nanmean(arr, axis=1)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(row_mean, inds[0])
    return arr
