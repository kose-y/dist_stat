from math import pi, inf, sqrt, nan
import torch
from .euclidean_distance import euclidean_distance_tensor
import time

def generate_grid(coord_x, coord_y):
    nx = coord_x.shape[0]
    ny = coord_y.shape[0]
    grid_x = coord_x.view(-1, 1).expand(-1, ny)
    grid_y = coord_y.expand(nx, -1)
    return torch.stack([grid_x, grid_y], dim=2).view(-1,2)

sparse_types = {torch.FloatTensor: torch.sparse.FloatTensor, 
                torch.DoubleTensor: torch.sparse.DoubleTensor, 
                torch.cuda.FloatTensor: torch.cuda.sparse.FloatTensor,
               torch.cuda.DoubleTensor: torch.cuda.sparse.DoubleTensor}

def E_generate(s=1, n_x=100, n_y=100, n_t=120, r = sqrt(2), dtype=torch.DoubleTensor, splits=10):
    x_min = -s
    x_max = s
    y_min = -s
    y_max = s

    # generate grid centers

    dx = (x_max - x_min)/n_x
    dy = (y_max - y_min)/n_y
    grid_x = x_min + dx * (torch.arange(n_x).type(dtype) + 0.5)
    grid_y = y_min + dy * (torch.arange(n_y).type(dtype) + 0.5)

    grid_xy = generate_grid(grid_x, grid_y) # center points

    # position of detector edges
    thetas = torch.stack([2*pi/n_t*torch.arange(n_t).type(dtype), 2*pi/n_t*(torch.arange(1, n_t+1).type(dtype))], dim=1) # denotes ranges
    detector_x = r * torch.cos(thetas)
    detector_y = r * torch.sin(thetas)
    detector_xy = torch.stack([detector_x[:, 0], detector_y[:, 0], detector_x[:, 1], detector_y[:,1]], dim=1) # boundary points, not centers



    # index pairs
    idx            = torch.arange(n_t).type(dtype).long()
    pairs_idx   = generate_grid(idx, idx)
    pairs_valid_ind = (pairs_idx[:, 0] < pairs_idx[:, 1]).view(-1,1)
    pairs_valid_idx = torch.masked_select(pairs_idx, pairs_valid_ind).view(-1,2)
    detector_pairs_xy = torch.cat([detector_xy[pairs_valid_idx[:, 0]], detector_xy[pairs_valid_idx[:, 1]]], dim=1)

    # distance between two detector edges
    intra_distance = sqrt(torch.sum((detector_xy[0, 0:2] - detector_xy[1, 2:4])**2)) 

    # distance between paired detectors
    # of course the distances are the same for dist_01 and dist_23.. these enhance numerical stability.
    dist_01 = torch.sqrt((detector_pairs_xy[:, 3] - detector_pairs_xy[:, 1])**2 + \
               (detector_pairs_xy[:, 2] - detector_pairs_xy[:, 0])**2).view(-1, 1) 
    dist_23 = torch.sqrt((detector_pairs_xy[:, 7] - detector_pairs_xy[:, 5])**2 + \
               (detector_pairs_xy[:, 6] - detector_pairs_xy[:, 4])**2).view(-1, 1)
    dist_12 = torch.sqrt((detector_pairs_xy[:, 5] - detector_pairs_xy[:, 3])**2 + \
               (detector_pairs_xy[:, 4] - detector_pairs_xy[:, 2])**2).view(-1, 1)
    dist_03 = torch.sqrt((detector_pairs_xy[:, 7] - detector_pairs_xy[:, 1])**2 + \
               (detector_pairs_xy[:, 6] - detector_pairs_xy[:, 0])**2).view(-1,1)

    # For each gridpt-detectorpair: compute if the point is inside the slice (mask out)
    assert detector_pairs_xy.shape[0] % splits == 0
    
    sz = detector_pairs_xy.shape[0] // splits
    mask_list = []
    for i in range(splits):
        print(i)
        rang = range(i * sz, (i + 1) * sz)
        
        m1 = ((detector_pairs_xy[rang, 7] - detector_pairs_xy[rang,1]).view(-1, 1) * \
            (grid_xy[:, 0] - detector_pairs_xy[rang, 2].view(-1, 1)) - \
            (detector_pairs_xy[rang, 6] - detector_pairs_xy[rang, 0]).view(-1, 1) * \
            (grid_xy[:, 1] - detector_pairs_xy[rang, 3].view(-1, 1)))
        m2 = ((detector_pairs_xy[rang, 7] - detector_pairs_xy[rang, 1]).view(-1, 1) * \
            (grid_xy[:, 0] - detector_pairs_xy[rang, 0].view(-1, 1)) - \
            (detector_pairs_xy[rang, 6] - detector_pairs_xy[rang, 0]).view(-1, 1) * \
            (grid_xy[:, 1] - detector_pairs_xy[rang, 1].view(-1, 1)))
        mask_part = to_sparse((m1*m2<=0))
        mask_list.append(mask_part)
    row_indices = [m._indices()[0,:].view(1, -1) + sz * i for (i,m) in enumerate(mask_list)]
    print(sz)
    print([m.shape for m in row_indices])
    print([m.min() for m in row_indices])
    print([m.max() for m in row_indices])
        
    mask_row_indices = torch.cat([m._indices()[0,:].view(1, -1) + sz * i for (i, m) in enumerate(mask_list)], dim=1)
    mask_col_indices = torch.cat([m._indices()[1,:].view(1, -1) for m in mask_list], dim=1)
    mask_values      = torch.cat([m._values() for m in mask_list])

    print(mask_row_indices.shape)
    print(mask_col_indices.shape)
    print(mask_values.shape)

    mask = torch.sparse.DoubleTensor(torch.cat([mask_row_indices, mask_col_indices], dim=0), mask_values, torch.Size([detector_pairs_xy.shape[0], grid_xy.shape[0]])).to(dtype=mask_values.dtype)
    
    print(mask._indices().shape)
    print(mask._nnz())
    print(mask.shape)

    # grid-detector distances
    detector_coord = detector_pairs_xy[mask._indices()[0, :], :] # 8 cols
    grid_coord = grid_xy[mask._indices()[1, :], :] # 2 cols
    print("detector pairs: ", detector_coord.shape)
    print("grid: ", grid_coord.shape)
    

    x_diff_0 = detector_coord[:, 0] - grid_coord[:, 0]
    y_diff_0 = detector_coord[:, 1] - grid_coord[:, 1]
    grid_detector_dist_0 = torch.sqrt(x_diff_0 ** 2 + y_diff_0 ** 2)

    x_diff_1 = detector_coord[:, 2] - grid_coord[:, 0]
    y_diff_1 = detector_coord[:, 3] - grid_coord[:, 1]
    grid_detector_dist_1 = torch.sqrt(x_diff_1 ** 2 + y_diff_1 ** 2)

    x_diff_2 = detector_coord[:, 4] - grid_coord[:, 0]
    y_diff_2 = detector_coord[:, 5] - grid_coord[:, 1]
    grid_detector_dist_2 = torch.sqrt(x_diff_2 ** 2 + y_diff_2 ** 2)

    x_diff_3 = detector_coord[:, 6] - grid_coord[:, 0]
    y_diff_3 = detector_coord[:, 7] - grid_coord[:, 1]
    grid_detector_dist_3 = torch.sqrt(x_diff_3 ** 2 + y_diff_3 ** 2)


    dist_01_ext = dist_01[mask._indices()[0,:]].view(-1)
    dist_23_ext = dist_23[mask._indices()[0,:]].view(-1)
    dist_12_ext = dist_12[mask._indices()[0,:]].view(-1)
    dist_03_ext = dist_03[mask._indices()[0,:]].view(-1)

    print(grid_detector_dist_0.shape)
    print(dist_01_ext.shape)

    # angle of views
    s_0 = (grid_detector_dist_0**2 + grid_detector_dist_1**2 - dist_01_ext**2)
    s_1 = (grid_detector_dist_2**2 + grid_detector_dist_3**2 - dist_23_ext**2)
    s_2 = (grid_detector_dist_0**2 + grid_detector_dist_3**2 - dist_03_ext**2)
    s_3 = (grid_detector_dist_1**2 + grid_detector_dist_2**2 - dist_12_ext**2)


    cos_0 = torch.clamp(s_0/(2*grid_detector_dist_0*grid_detector_dist_1), min=-1, max=1) 
    cos_1 = torch.clamp(s_1/(2*grid_detector_dist_2*grid_detector_dist_3), min=-1, max=1)
    cos_2 = torch.clamp(s_2/(2*grid_detector_dist_0*grid_detector_dist_3), min=-1, max=1)
    cos_3 = torch.clamp(s_3/(2*grid_detector_dist_1*grid_detector_dist_2), min=-1, max=1)

    angle_0 = torch.acos(cos_0)
    angle_1 = torch.acos(cos_1)
    angle_2 = pi - torch.acos(cos_2)
    angle_3 = pi - torch.acos(cos_3)

    angles = torch.stack([angle_0, angle_1, angle_2, angle_3], dim=1)

    angle = torch.min(angles, dim=1)[0]

    r = sparse_types[dtype](mask._indices(), angle/pi, mask.shape) # make it sparse?
    
    return r


def grid_matrix(n_x=100, n_y=100, dtype=torch.DoubleTensor):
    x_coords = torch.arange(n_x).long()
    y_coords = torch.arange(n_y).long()
    coords = generate_grid(x_coords, y_coords)
    indices = torch.arange(coords.shape[0]).long()
    
    pairs_x = torch.stack([indices[:-n_y], indices[n_y:]], dim=1)
    pairs_y = torch.stack([indices[:-1], indices[1:]], dim=1)

    select_x = (torch.sum(coords[pairs_x[:, 1]] - coords[pairs_x[:, 0]] == torch.LongTensor([1,0 ]), dim=1) == 2).view(-1, 1)
    select_y = (torch.sum(coords[pairs_y[:, 1]] - coords[pairs_y[:, 0]] == torch.LongTensor([0, 1]), dim=1) == 2).view(-1, 1)
    
    pairs_x_filtered = torch.masked_select(pairs_x, select_x).view(-1, 2)
    pairs_y_filtered = torch.masked_select(pairs_y, select_y).view(-1, 2)
    
    all_edges = torch.cat([pairs_x_filtered, pairs_y_filtered], dim=0)
    v = torch.Tensor(all_edges.shape[0]).fill_(1).type(dtype)
    g = sparse_types[dtype](all_edges.t(), v, torch.Size([coords.shape[0], coords.shape[0]]))
    g = g + g.t()
    
    edge_indices = torch.arange(all_edges.shape[0]).long()
    pos = torch.stack([edge_indices, all_edges[:, 0]], dim=0)
    neg = torch.stack([edge_indices, all_edges[:, 1]], dim=0)
    i = torch.cat([pos, neg], dim=1)
    ones = torch.Tensor(pos.shape[1]).fill_(1).type(dtype)
    v = torch.cat([ones, -ones])
    d = sparse_types[dtype](i, v, torch.Size([pos.shape[1], indices.shape[0]]))

    return g, d

def data_generate(inputimg, emat, num_samples=100000, batch_size=10000):
    assert num_samples%batch_size==0
    #emat = emat.type(TType)
    #TType = emat.type()
    #emat_typename = torch.typename(emat).split('.')[-1]
    sparse_tensortype = emat.type()
    #emat = emat.coalesce()
    import scipy.sparse as sp
    emat_np_ind = emat._indices().to(device='cpu').numpy()
    emat_np_val = emat._values().to(device='cpu').numpy()
    print(emat_np_ind.shape)
    print(emat_np_val.shape)
    emat_sp = sp.coo_matrix((emat_np_val, (emat_np_ind[0,:], emat_np_ind[1,:])), shape=(emat.shape[0], emat.shape[1])).tocsc()
     

    num_batches = num_samples//batch_size
    inputimg = inputimg.view(-1)
    pxl = torch.multinomial(inputimg, num_samples, replacement=True)
    count = torch.zeros(emat.shape[0]).to(dtype=emat.dtype)
    one_ttype = torch.ones(batch_size).to(dtype=emat.dtype)
    tmp_storage = torch.zeros(batch_size, emat.shape[0]).to(dtype=emat.dtype)
    for i in range(num_batches):
        print(i)
        #batch_probs = emat.index_select(1, pxl[(i*batch_size):((i+1)*batch_size)]).t()
        batch_probs = torch.tensor(emat_sp[:, pxl[(i*batch_size):((i+1)*batch_size)]].T.todense())
        detector_pair = torch.multinomial(batch_probs, 1, replacement=True).view(-1)
        count.scatter_add_(0, detector_pair, one_ttype)
    return count


# taken from: https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/3
def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
