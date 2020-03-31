import numpy as np
from scipy import spatial

def convert_functional_map_to_pointwise_map(C12, B1, B2):
    '''
    Pointwise map reconstruction
    :param C12: given functional map C12: S1 -> S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :return: T21: the pointwise map T21: S2 -> S1 (index 0-based)
    '''
    if C12.shape[0] != B2.shape[1] or C12.shape[1] != B1.shape[1]:
        return -1
    else:
        _, T21 = spatial.cKDTree(np.matmul(B1, C12.transpose())).query(B2, n_jobs=-1)
        return T21

def convert_pointwise_map_to_functional_map(T12, B1, B2):
    '''
    Convert a pointwise map to a functional map
    :param T12: given pointwise map T12: S1 -> S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :return: C21: the corresponding functional map C21: S2 -> S1
    '''
    C21 = np.linalg.lstsq(B1, B2[T12, :], rcond=None)[0]
    return C21

def refine_pMap_icp(T12, B1, B2, num_iters=10):
    '''
    Regular Iterative Closest Point (ICP) to refine a pointwise map
    :param T12: initial pointwise map from S1 to S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :param num_iters: the number of iterations for refinement
    :return: T12_refined, C21_refined
    '''
    T12_refined = T12
    for i in range(num_iters):
        C21_refined = convert_pointwise_map_to_functional_map(T12_refined, B1, B2)
        T12_refined = convert_functional_map_to_pointwise_map(C21_refined, B2, B1)
    return T12_refined, C21_refined

def refine_pMap_zo(T12, B1, B2, start, num_iters=10):
    '''
    Regular Iterative Closest Point (ICP) to refine a pointwise map
    :param T12: initial pointwise map from S1 to S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :param num_iters: the number of iterations for refinement
    :return: T12_refined, C21_refined
    '''
    
    neig_start = start
    neig_end = B1.shape[1]
    up_step = (neig_end - neig_start)/num_iters
    up_inds = np.arange(neig_start, neig_end + up_step, up_step).astype(int)
    #print(up_inds)
    
    T12_refined = T12
    #last_ind = neig_start
    for i in up_inds:
        #print(i)
        C21_refined = convert_pointwise_map_to_functional_map(T12_refined, B1[:, :i], B2[:, :i])
        #C12 size (lastind, lastind)
        T12_refined = convert_functional_map_to_pointwise_map(C21_refined, B2[:, :i], B1[:, :i])
        #print(C21_refined.shape)
        #print(T12_refined)
        #last_ind = i
    return T12_refined, C21_refined

def viz_textmap(v):
    minv = np.min(v, axis = 0)
    maxv = np.max(v, axis = 0)
    
    uv1 = (v[:, 0] - minv[0])#/(maxv[0] - minv[0])
    uv2 = (v[:, 1] - minv[1])#/(maxv[1] - minv[1])
    #print(np.min(uv1), np.max(uv1))
    #print(np.min(uv2), np.max(uv2))
    
    return np.stack([uv1, uv2], axis = -1)
