import numpy as np
import scipy as sp
import scipy.linalg

def tmatrix_e0(diag, offdiag):
    """ Compute ground state energy of tridiagonal matrix

    Args:
         diag     :  np.array with diagonal elements, length N
         offdiag  :  np.array with offdiagonal elements, length N-1
    Returns:
         float    :  ground state energy (smallest eigenvalue)
    """
    if len(diag) == 1:
        return diag[0]
    else:
        return sp.linalg.eigvalsh_tridiagonal(diag, offdiag, 'i', 
                                              select_range=(0,0),
                                              check_finite=False)[0]
    # return sp.linalg.eigvalsh_tridiagonal(diag, offdiag, 'a')[0]


def tmatrix_eigvals(diag, offdiag):
    """ Compute all eigenvalues of tridiagonal matrix

    Args:
         diag     :  np.array with diagonal elements, length N
         offdiag  :  np.array with offdiagonal elements, length N-1
    Returns:
         np.array :  all eigenvalues
    """
    if len(diag) == 1:
        return diag
    else:
        return sp.linalg.eigvalsh_tridiagonal(diag, offdiag,
                                              check_finite=False, select='a')


def tmatrix_eig(diag, offdiag):
    """ Compute all eigenvalues / eigenvectors of tridiagonal matrix

    Args:
         diag     :  np.array with diagonal elements, length N
         offdiag  :  np.array with offdiagonal elements, length N-1
    Returns:
         np.array, np.array :  all eigenvalues, all eigenvectors
    """
    if len(diag) == 1:
        return diag, np.ones((1,1))
    else:
        return sp.linalg.eigh_tridiagonal(diag, offdiag,
                                          check_finite=False, select='a',
                                          lapack_driver='stev')


def boltzmann_tensor(eigs, e0, betas, shifts=None, check_posdef=True):
    """ Compute the Boltzmann factor tensor

    If NO shifts mu_k are defined, the Boltzmannn factor tensor is a 2D tensor, 
    
    B_ij = \exp ( -\beta_j ( e_i - e_0 ) )           


    If shifts mu_k are defined, the Boltzmannn factor tensor is a 3D tensor, 
    
    B_ijk = \exp ( -\beta_j ( e_i - e_0 + \mu_k ) )           


    Args:
        eigs     : np.array, eigenvalues e_i
        e0       : float of np.array, minimal eigenvalue offset (for each shift)
        shifts   : np.array, shifts mu_k
        check_posdef : check whether all weights are positive in exponentiation
    Returns:
        np.array : Boltzmann tensor
    """
    if shifts is None:
        eigs_e0 = np.outer(eigs - e0, betas)
        if check_posdef and not np.all(eigs_e0) > -1e-12:
            raise ValueError("eigs - e0 not always positive")
        # print("eigs - e0", eigs - e0)
        # print("beigs[0]", eigs[0])
        # print("be0", e0)
        # print("-eigs_e0", -eigs_e0)
        tensor = np.exp(-eigs_e0)
    else:
        if shifts.shape != e0.shape:
            raise ValueError("Incompatible shifts and e0 shape")
        eigs_shift_e0 = np.add.outer(eigs, shifts - e0)
        if check_posdef and not np.all(eigs_shift_e0) > -1e-12:
            raise ValueError("eigs - e0 + shifts not always positive")

        tensor = np.einsum("i,jk->jik", -betas, eigs_shift_e0, optimize="optimal")
        tensor = np.exp(tensor)

    return tensor


def moment_average(diag, offdiag, e0, betas, shifts=None, k=0, check_posdef=True):
    """ Compute a moment average of a tridiagonal matrix of the form

    M_{ij}^{(k)}= e_0 (T+\mu_j)^k \exp( - \beta_i (T - e0 + \mu_j) e_0

    Args:
         diag     :  np.array with diagonal elements, length N
         offdiag  :  np.array with offdiagonal elements, length N-1
         e0       : float of np.array, minimal eigenvalue offset (for each shift)
         shifts   : np.array, shifts mu_k
         check_posdef : check whether all weights are positive in exponentiation

    Returns:
        np.array : moment average tensor, M_{ij}^{(k)} as defined above
    """
    eigs, U = tmatrix_eig(diag, offdiag)
    U_dag = np.transpose(U.conj())

    b_tensor = boltzmann_tensor(eigs, e0, betas, shifts, check_posdef)
    if shifts is None:
        moment_tensor = np.einsum("i,ij->ij", np.power(eigs, k), b_tensor, 
                                  optimize="optimal")
        tensor_U_dag0 = np.einsum("ij, i-> ij", moment_tensor, U_dag[:,0], 
                                  optimize="optimal")
        avg = np.einsum("i, ij -> j", U[0,:], tensor_U_dag0, optimize="optimal")
        # print("avg", avg)

        # print("inner avg ", np.array(avg)[:3])
        # tmat = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
        # eigs = np.linalg.eigvalsh(tmat)
        # tmat0 = tmat - eigs[0]*np.diag(np.ones_like(diag))

        # avg2 = []
        # zzs = []
        # for idx, beta in enumerate(betas):
        #     expm = scipy.linalg.expm(-beta/2 * tmat0)
        #     mm = np.dot(expm, np.dot(np.linalg.matrix_power(tmat0, k), expm))
        #     avg2.append(mm[0,0])
        #     zz = np.dot(expm, expm)
        #     zzs.append(zz[0,0])
        # avg2 = np.array(avg2)
        # zzs = np.array(zzs)
        # # print("avg2", avg2)
        # avg2e = avg2 * np.exp(betas * (e0 - eigs[0]))
        # print("avg2e", avg2e)
        # print("eigs[0]", eigs[0], "e0", e0)
        # assert(np.allclose(avg, avg2e))
        
    else:
        eigs_shifted = np.add.outer(eigs, shifts)
        moment_tensor = np.einsum("ik,ijk->ijk", np.power(eigs_shifted, k),
                                  b_tensor, optimize="optimal")
        tensor_U_dag0 = np.einsum("ijk, i-> ijk", moment_tensor, U_dag[:,0], 
                                  optimize="optimal")
        avg = np.einsum("i, ijk -> jk", U[0,:], tensor_U_dag0, optimize="optimal")

    return avg


def operator_average(diag, offdiag, operator, e0, betas, shifts=None, k=1, 
                     check_posdef=True):
    """ Compute an operator average of a tridiagonal matrix of the form

    (A)_{ij}^{(k)}= e_0 \exp( - \beta_i/2 (T - e0 + \mu_j) A \exp( - \beta_i/2 (T - e0 + \mu_j) e_0

    Args:
         diag     :  np.array with diagonal elements, length N
         offdiag  :  np.array with offdiagonal elements, length N-1
         e0       : float of np.array, minimal eigenvalue offset (for each shift)
         shifts   : np.array, shifts mu_k
         check_posdef : check whether all weights are positive in exponentiation

    Returns:
        np.array : moment average tensor, M_{ij}^{(k)} as defined above
    """
    eigs, U = tmatrix_eig(diag, offdiag)
    U_dag = np.transpose(U.conj())

    operator_adj = np.dot(U_dag, np.dot(np.linalg.matrix_power(operator, k), U))
    b_tensor = boltzmann_tensor(eigs, e0, betas / 2. , shifts, check_posdef)
    # print("eigs", eigs[:3])
    # print("oper", operator[:5,:5])
    # tmat = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
    # print("tmat", tmat[:5,:5])    
    # print("diag", diag[:3])
    # print("offd", offdiag[:3])
        
    if shifts is None:
        l_tensor = np.einsum("i, ij -> ij", U[0,:], b_tensor, optimize="optimal")
        r_tensor = np.einsum("ij, i -> ij", b_tensor, U_dag[:,0], optimize="optimal")
        avg = np.einsum("aj, ab, bj -> j", l_tensor, operator_adj, r_tensor, optimize="optimal")

        # # print("inner avg ", np.array(avg)[:3])
        # avg2 = []
        # zzs = []
        # for idx, beta in enumerate(betas):

        #     eigs = np.linalg.eigvalsh(tmat)
        #     tmat0 = tmat - eigs[0]*np.diag(np.ones_like(diag))
        #     expm = scipy.linalg.expm(-beta/2 * tmat0)
        #     mm = np.dot(expm, np.dot(operator, expm))
        #     avg2.append(mm[0,0])
        #     zz = np.dot(expm, expm)
        #     zzs.append(zz[0,0])
        # avg2 = np.array(avg2)
        # zzs = np.array(zzs)
        # print("inner avg2", avg2[:3])
        # print("inner part", zzs[:3])
        # print("inner expt", (avg2 / zzs)[:3])
    else:
        b_tensor_l = np.einsum("i, ijk -> ijk", U[0,:], b_tensor,
                                 optimize="optimal")

        b_tensor_r = np.einsum("ijk, i -> ijk", b_tensor, U_dag[:,0],
                               optimize="optimal")

        avg = np.einsum("ajk, ab, bjk -> jk",
                        b_tensor_l, operator_adj, b_tensor_r,
                        optimize="optimal")
     
    return avg

