import numpy as np
from . import utils
def default_partitioner(n, nd):
    q = n // nd
    r = n % nd
    nis = [q+(1 if i<r else 0) for i in range(nd)]
    return np.cumsum([0]+nis)

def _group_partitioner(n, nd, gpart):
    part = default_partitioner(n,nd)
    for i in range(len(part)):
        part[i] = utils.find_nearest(gpart, part[i])
    return part
def group_partitioner(gpart):
    return lambda n, nd: _group_partitioner(n, nd, gpart)
   
def _groupvar_partitioner(n, nd, ipart, gpart):
    assert n == len(gpart)-1
    part = [0]
    for i in range(nd):
        grp_bd = int(np.where(gpart == utils.find_nearest(gpart, ipart[i+1]))[0])
        part.append(grp_bd)
    return part
def groupvar_partitioner(ipart, gpart):
    return lambda n, nd: _groupvar_partitioner(n, nd, ipart, gpart)
   
    


