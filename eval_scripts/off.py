import numpy as np
from scipy.sparse import csr_matrix, spdiags
import time


def readOFF(file):
    file = open(file, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')

    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    vertst = np.array(verts).T.tolist()  # colors ?

    return np.array(verts), np.array(faces)

#####################
### Spectral Data ###
#####################

def cotangent_laplacian(V, T):
    '''
    Compute the cotangent matrix and the weight matrix of a shape
    :param S: a shape with VERT and TRIV
    :return: cotangent matrix W, and the weight matrix A
    '''

    nv = V.shape[0]

    T1 = T[:, 0]
    T2 = T[:, 1]
    T3 = T[:, 2]

    V1 = V[T1, :]
    V2 = V[T2, :]
    V3 = V[T3, :]


    L1 = np.linalg.norm(V2 - V3, axis=1)
    L2 = np.linalg.norm(V1 - V3, axis=1)
    L3 = np.linalg.norm(V1 - V2, axis=1)
    L = np.column_stack((L1, L2, L3))  # Edges of each triangle

    Cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
    Cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
    Cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)
    Cos = np.column_stack((Cos1, Cos2, Cos3))  # Cosines of opposite edges for each triangle
    Ang = np.arccos(Cos)  # Angles

    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))
    w = 0.5 * cotangent(np.concatenate((Ang[:, 2], Ang[:, 0], Ang[:, 1]))).astype(float)
    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    wn = np.concatenate((-w, -w, w, w))
    W = csr_matrix((wn, (In, Jn)), [nv, nv])  # Sparse Cotangent Weight Matrix

    cA = cotangent(Ang) / 2  # Half cotangent of all angles
    At = 1 / 4 * (L[:, [1, 2, 0]] ** 2 * cA[:, [1, 2, 0]] + L[:, [2, 0, 1]] ** 2 * cA[:, [2, 0, 1]]).astype(
        float)  # Voronoi Area

    # TODO (to update): the MATLAB version gives the area of the parallelogram, not the triangle
    # Ar = 0.5*np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1)  # todo - correct version
    Ar = np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1)  # todo - Matlab version

    # Use Ar is ever cot is negative instead of At
    locs = cA[:, 0] < 0
    At[locs, 0] = Ar[locs] / 4
    At[locs, 1] = Ar[locs] / 8
    At[locs, 2] = Ar[locs] / 8

    locs = cA[:, 1] < 0
    At[locs, 0] = Ar[locs] / 8
    At[locs, 1] = Ar[locs] / 4
    At[locs, 2] = Ar[locs] / 8

    locs = cA[:, 2] < 0
    At[locs, 0] = Ar[locs] / 8
    At[locs, 1] = Ar[locs] / 8
    At[locs, 2] = Ar[locs] / 4

    Jn = np.zeros(I.shape[0])
    An = np.concatenate((At[:, 0], At[:, 1], At[:, 2]))
    Area = csr_matrix((An, (I, Jn)), [nv, 1])  # Sparse Vector of Area Weights

    In = np.arange(nv)
    A = csr_matrix((np.squeeze(np.array(Area.todense())), (In, In)),
                   [nv, nv])  # Sparse Matrix of Area Weights
    return W, A


def cotangent(p):
    return np.cos(p) / np.sin(p)


def compute_vertex_and_face_normals(V, F):
    '''
    Compute the per-face and per-vertex mesh normals
    :param S: the input triangle mesh
    :return: the per-face normals Nf, and the per-vertex normals Nv
    '''
    #F = S.TRIV
    #V = S.VERT
    num_face = F.shape[0]
    num_vtx = V.shape[0]
    # per-face normal vector
    Nf = np.cross(V[F[:, 1], :] - V[F[:, 0], :], V[F[:, 2], :] - V[F[:, 1], :])
    Fa = 0.5 * np.sqrt(np.power(Nf, 2).sum(axis=1))

    # normalize each vector to unit length
    for i in range(num_face):
        Nf[i, :] = np.divide(Nf[i, :], np.sqrt(np.sum(np.power(Nf[i, :], 2))))

    # per-vertex normal vector
    Nv = np.zeros(V.shape)
    for i in range(num_face):
        for j in range(3):
            if j == 0:
                la = np.sum(np.power(V[F[i, 0], :] - V[F[i, 1], :], 2))
                lb = np.sum(np.power(V[F[i, 0], :] - V[F[i, 2], :], 2))
                W = Fa[i] / (la * lb)
            elif j == 1:
                la = np.sum(np.power(V[F[i, 1], :] - V[F[i, 0], :], 2))
                lb = np.sum(np.power(V[F[i, 1], :] - V[F[i, 2], :], 2))
                W = Fa[i] / (la * lb)
            else:
                la = np.sum(np.power(V[F[i, 2], :] - V[F[i, 0], :], 2))
                lb = np.sum(np.power(V[F[i, 2], :] - V[F[i, 1], :], 2))
                W = Fa[i] / (la * lb)

            if np.isinf(W) or np.isnan(W):
                W = 0

            Nv[F[i, j], :] = Nv[F[i, j], :] + Nf[i, :] * W

    # normalize each vector to unit length
    for i in range(num_vtx):
        Nv[i, :] = np.divide(Nv[i, :], np.sqrt(np.sum(np.power(Nv[i, :], 2))))

    return Nv, Nf


# --------------------------------------------------------------------------
#  functions for constructing orientation operators (by Adrien Poulenard)
# --------------------------------------------------------------------------


def compute_face_area(V, T):
    '''
    Compute the area for each triangle face
    :param S: given mesh
    :return: per-face area (nf-by-1) vector
    '''
    #T = S.TRIV
    #V = S.VERT

    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    # TODO: (to update) the MATLAB version is wrong
    # Ar = 0.5 * np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1) # todo - correct version
    Ar = 0.5 * np.sum(np.power(np.cross(V1 - V2, V1 - V3), 2), axis=1)  # todo - MATLAB version
    return Ar


def compute_mesh_surface_area(V, T):
    '''
    Compute the surface area of a mesh
    :param S: given mesh
    :return: surface area
    '''
    return sum(compute_face_area(V, T))


def mass_matrix(V, T):
    '''
    Compute the mass matrix (the construction is slightly different from that in cotangent_laplacian)
    This can be replaced by simply using S.A
    :param S: given mesh
    :return: return the mass matrix (sparse)
    '''

    #T = S.TRIV
    #V = S.VERT
    nv = V.shape[0]

    Ar = compute_face_area(V, T)

    T1 = T[:, 0]
    T2 = T[:, 1]
    T3 = T[:, 2]

    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))
    Mij = (1 / 12) * np.concatenate((Ar, Ar, Ar))
    Mji = Mij
    Mii = (1 / 6) * np.concatenate((Ar, Ar, Ar))
    In = np.concatenate((I, J, I))
    Jn = np.concatenate((J, I, I))
    Mn = np.concatenate((Mij, Mji, Mii))

    M = csr_matrix((Mn, (In, Jn)), [nv, nv])  # Sparse Cotangent Weight Matrix
    return M


def compute_function_grad_on_faces(V, T, f):
    '''
    Compute the gradient of a function on the faces
    :param S: given mesh
    :param f: a function defined on the vertices
    :return: the gradient of f on the faces of S
    '''

    _, Nf = compute_vertex_and_face_normals(V, T)

    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    Ar = compute_face_area(V, T)
    Ar = np.tile(np.array(Ar, ndmin=2).T, (1, 3))

    f = np.array(f, ndmin=2).T  # make sure the function f is a column vector
    grad_f = np.divide(np.multiply(np.tile(f[T[:, 0]], (1, 3)), np.cross(Nf, V3 - V2)) +
                       np.multiply(np.tile(f[T[:, 1]], (1, 3)), np.cross(Nf, V1 - V3)) +
                       np.multiply(np.tile(f[T[:, 2]], (1, 3)), np.cross(Nf, V2 - V1))
                       , 2 * Ar)
    return grad_f


def vector_field_to_operator(V, T, Vf):
    '''
    Convert a vector field to an operator
    :param S: given mesh
    :param Vf: a given tangent vector field
    :return: an operator "equivalent" to the vector field
    '''

    nv = V.shape[0]

    _, Nf = compute_vertex_and_face_normals(V, T)

    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    Jc1 = np.cross(Nf, V3 - V2)
    Jc2 = np.cross(Nf, V1 - V3)
    Jc3 = np.cross(Nf, V2 - V1)

    T1 = T[:, 0]
    T2 = T[:, 1]
    T3 = T[:, 2]
    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))

    Sij = 1 / 6 * np.concatenate((np.sum(np.multiply(Jc2, Vf), axis=1),
                                  np.sum(np.multiply(Jc3, Vf), axis=1),
                                  np.sum(np.multiply(Jc1, Vf), axis=1)))

    Sji = 1 / 6 * np.concatenate((np.sum(np.multiply(Jc1, Vf), axis=1),
                                  np.sum(np.multiply(Jc2, Vf), axis=1),
                                  np.sum(np.multiply(Jc3, Vf), axis=1)))

    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    Sn = np.concatenate((Sij, Sji, -Sij, -Sji))
    W = csr_matrix((Sn, (In, Jn)), [nv, nv])
    M = mass_matrix(V, T)
    tmp = spdiags(np.divide(1, np.sum(M, axis=1)).T, 0, nv, nv)

    op = tmp @ W
    return op

import matplotlib.pyplot as plt
def compute_orientation_operator_from_a_descriptor(V, T, B, f):
    '''
    Extract the orientation information from a given descriptor
    :param S: given mesh
    :param B: the basis (should be consistent with the fMap)
    :param f: a descriptor defined on the mesh
    :return: the orientation operator (preserved via commutativity by a fMap)
    '''

    _, Nf = compute_vertex_and_face_normals(V, T)

    # normalize the gradient to unit length
    grad_f = compute_function_grad_on_faces(V, T, f)
    length = np.sqrt(np.sum(np.power(grad_f, 2), axis=1)) + 1e-16
    tmp = np.tile(np.array(length, ndmin=2).T, (1, 3))
    norm_grad = np.divide(grad_f, tmp)

    # rotate the gradient by pi/2
    rot_norm_grad = np.cross(Nf, norm_grad)

    # TODO: check which operator should be used
    # Op = vector_field_to_operator(S, norm_grad)
    # diff_Op = np.matmul(B.transpose(), np.matmul(S.A.toarray(), np.matmul(Op, B)))

    # convert vector field to operators
    Op_rot = vector_field_to_operator(V, T, rot_norm_grad)

    # create 1st order differential operators associated with the vector fields
    _, A = cotangent_laplacian(V, T)
    #print(B.transpose().shape, A.shape, Op_rot.shape, B.shape)
    #plt.imshow(Op_rot.toarray())
    #plt.colorbar()
    #plt.show()
    Op_rot = Op_rot.todense()
    A = A.todense()
    diff_Op_rot = np.matmul(B.transpose(),
                            np.matmul(A,
                                      Op_rot @ B))

    return diff_Op_rot

# --------------------------------------------------------------------------
#  functions for constructing orientation operators  - End
# --------------------------------------------------------------------------