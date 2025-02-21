import numpy as np
import scipy as sp

from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import norm as sp_norm, spsolve
from scipy.sparse import triu as sp_triu


def arnoldi(A, V, k, precondition=False, M=None):
    """
    Computes one iteration of Arnoldi iteration given the iteration index k

    :param precondition:
    :param A:
    :param V:
    :param k:
    :return:
    """

    m, _ = A.shape

    # inialize k + 1 nonzero elements of H along column k
    # k starts at 0...
    h_k = np.zeros((k + 2, ))

    if precondition:
        # w = spsolve(M, A @ V[:, k])

        L, U = M
        z = spsolve(L, A @ V[:, k])
        w = spsolve(U, z)
    else:
        w = A @ V[:, k]

    # calculate first k elements of the kth Hessenberg column
    for i in range(k + 1):
        h_k[i] = w @ V[:, i]
        w = w - h_k[i] * V[:, i]

    # h_k[k + 1] = sp_norm(w)
    h_k[k + 1] = np_norm(w)

    # check if 0
    if h_k[k + 1] == 0:
        return h_k, None
    else:
        # assert h_k[k + 1] != 0
        v_new = w / h_k[k + 1]

    # v_new = w / h_k[k + 1]

    return h_k, v_new





