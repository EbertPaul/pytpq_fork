import numpy as np
import scipy as sp

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


def boltzmann_tensor(eigs, e0, betas, check_posdef=True):
    """ Compute the Boltzmann factor tensor

    Boltzmannn factor tensor is a 2D tensor, 
    
    B_ij = \exp ( -\beta_j ( e_i - e_0 ) )           
       
    Args:
        eigs     : np.array, eigenvalues e_i
        e0       : float of np.array, minimal eigenvalue offset
        check_posdef : check whether all weights are positive in exponentiation
    Returns:
        np.array : Boltzmann tensor
    """
    eigs_e0 = np.outer(eigs - e0, betas)
    if check_posdef and not np.all(eigs_e0) > -1e-12:
        raise ValueError("eigs - e0 not always positive")
    # print("eigs - e0", eigs - e0)
    # print("beigs[0]", eigs[0])
    # print("be0", e0)
    # print("-eigs_e0", -eigs_e0)
    tensor = np.exp(-eigs_e0)
    return tensor


def moment_average(diag, offdiag, e0, betas, k=0, check_posdef=True):
    """ Compute a moment average of a tridiagonal matrix of the form

    M_{ij}^{(k)}= e_0 (T)^k \exp( - \beta_i (T - e0) e_0

    Args:
         diag     :  np.array with diagonal elements, length N
         offdiag  :  np.array with offdiagonal elements, length N-1
         e0       : float of np.array, minimal eigenvalue offset
         check_posdef : check whether all weights are positive in exponentiation

    Returns:
        np.array : moment average tensor, M_{ij}^{(k)} as defined above
    """
    eigs, U = tmatrix_eig(diag, offdiag)
    U_dag = np.transpose(U.conj())
    b_tensor = boltzmann_tensor(eigs, e0, betas, check_posdef)
    moment_tensor = np.einsum("i,ij->ij", np.power(eigs, k), b_tensor, 
                                optimize="optimal")
    # transforming e_0 to the T eigenbasis produces U_dag[:,0] from the right and U[:,0] from the left
    tensor_U_dag0 = np.einsum("ij, i-> ij", moment_tensor, U_dag[:,0], optimize="optimal")
    avg = np.einsum("i, ij -> j", U[0,:], tensor_U_dag0, optimize="optimal")
    return avg
