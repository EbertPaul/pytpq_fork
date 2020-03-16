import numpy as np
import numexpr as ne
import pytpq.linalg as pla
from collections import OrderedDict
from scipy.special import binom
import pytpq.basic as pba
import pytpq.linalg as pla
import pytpq.thermo as pth

import time


def eigs_poles_weights_infty(diag_left, offdiag_left, 
                             diag_right, offdiag_right, 
                             operator, rV, VAr):
    evals_left, evecs_left = pla.tmatrix_eig(diag_left, offdiag_left)
    evals_right, evecs_right = pla.tmatrix_eig(diag_right, offdiag_right)

    poles = -np.subtract.outer(evals_left, evals_right)

    operator_adj = np.dot(np.dot(evecs_left.T, operator), evecs_right)
    w_left = np.dot(rV, evecs_left)
    w_right = np.dot(evecs_right.T, VAr)
    weights = np.einsum("k, kl, l -> kl", w_left, operator_adj, w_right, 
                        optimize="optimal")
    
    return evals_left, poles, weights


def dynamic_spectra(ensemble, temperatures, shifts=None, e0=None,  
                    alpha_tag="AlphasV", 
                    beta_tag="BetasV",
                    alpha_tilde_tag="AlphasVTilde", 
                    beta_tilde_tag="BetasVTilde",
                    operator_tag="VsAVsTilde",
                    lanczos_overlap_tag="RDotVs", 
                    lanczos_operator_overlap_tag="VsTildeDotAR",
                    crop=True, check_posdef=True):

    # Get inverse temperatures and ground state
    if not np.all(temperatures) > 0:
        raise ValueError("Invalid temperatures")
    temperatures = np.array(temperatures)
    n_temperatures = len(temperatures)

    betas = 1. / temperatures    
    if e0 == None:
        e0, _ = pba.ground_state_energy(ensemble, shifts=shifts)

    partitions = pth.moment_sum(ensemble, temperatures, shifts, 0, e0,  
                                alpha_tag, beta_tag, crop, check_posdef)

    pole_list = OrderedDict()
    weight_list = OrderedDict()

    for seed in ensemble.seeds:
        
        # Initialize pole list and weight list for this seed
        pole_list[seed] = np.empty((0,))
        if shifts is None:
            weight_list[seed] = np.empty((0, n_temperatures))
        else:
            n_shifts = len(shifts[ensemble.seeds[0]][ensemble.qns[0]])
            weight_list[seed] = np.empty((0, n_temperatures, n_shifts))

        for qn in ensemble.qns:
            
            # get pole positions and T=infty weights
            diag_left, offdiag_left = pba.tmatrix(ensemble, seed, qn, alpha_tag, 
                                              beta_tag, crop=crop)
            diag_right, offdiag_right = pba.tmatrix(ensemble, seed, qn, alpha_tilde_tag, beta_tilde_tag)

            size_left = len(diag_left)
            size_right = len(diag_right)
            n_poles = size_left * size_right

            operator = ensemble.data(seed, qn, operator_tag)[:size_left, :size_right]
            rV = ensemble.data(seed, qn, lanczos_overlap_tag)[:size_left]
            VAr = ensemble.data(seed, qn, lanczos_operator_overlap_tag)[:size_right]

            eigs, poles, weights_infty = \
                eigs_poles_weights_infty(diag_left, offdiag_left, 
                                         diag_right, offdiag_right, 
                                         operator, rV, VAr)

            poles = np.reshape(poles, (n_poles))
            dim = ensemble.dimension[qn]

            # get weights at finite temperature
            if shifts is None:
                b_tensor = pla.boltzmann_tensor(eigs, e0[seed], betas, None)
                weights = dim * np.einsum("mj, mn -> mnj", b_tensor, weights_infty) / partitions[seed]
  
                # # Check poles, weights
                # es, evecs_left = pla.tmatrix_eig(diag_left, offdiag_left)
                # est, evecs_right = pla.tmatrix_eig(diag_right, offdiag_right)

                # psis = np.dot(rV, evecs_left)
                # psist = np.dot(evecs_right.T, VAr)
                # Apsi = np.dot(np.dot(evecs_left.T, operator), evecs_right)
                # beta=1.
                
                # wgt = np.zeros((size_left, size_right))

                # for i in range(size_left):
                #     for j in range(size_right):
                #         pole = est[j] - es[i] 
                #         wgt[i,j] = dim * psis[i].conj() * Apsi[i,j].conj() * psist[j].conj() * np.exp(-beta * (es[i] - e0[seed]))  / partitions[seed]

                # print(wgt[:4,:4])
                # print(weights[:4,:4,0])
                # print()
                # assert(np.allclose(wgt, weights[:,:,0]))

                weights = np.reshape(weights, (n_poles, n_temperatures))
            else:
                if not len(shifts[seed][qn]) == n_shifts:
                    raise ValueError("Not all shift arrays are equally long")
                b_tensor = pla.boltzmann_tensor(eigs, e0[seed], betas, shifts[seed][qn])
                weights = dim * np.einsum("mjk, mn -> mnjk", b_tensor, weights_infty) / partitions[seed]
                weights = np.reshape(weights, (n_poles, n_temperatures, n_shifts))

            pole_list[seed] = np.append(pole_list[seed], poles, axis=0)
            weight_list[seed] = np.append(weight_list[seed], weights, axis=0)
            assert pole_list[seed].shape[0] == weight_list[seed].shape[0] 


                    # weight = psis[i].conj() * Apsi[i,j].conj() * psist[j].conj() * np.exp(-beta * (es[i] - e0)) / partition
                    # print(pole, poles)
            # print(poles.shape)
            # print(pole_list.shape)
            # print(weights.shape)
            # print(weight_list.shape)
            # print()
            
    return pole_list, weight_list

def broaden(omegas, poles, weights, eta):

    # t0 = time.time()
    diffs = np.subtract.outer(omegas, poles)
    gaussian_tensor = ne.evaluate('exp(-0.5 * (diffs / eta)**2)')
    gaussian_tensor *= (1. / (eta * np.sqrt(2*np.pi))) 

    # t1 = time.time()
    # print("numexpr", t1-t0, "secs")

    # t0 = time.time()
    # gaussian_tensor2 = -0.5*(np.subtract.outer(omegas, poles) / eta)**2
    # t1 = time.time()
    # print("sub", t1-t0, "secs")
    
    # t0 = time.time()
    # # fast numexpr exponentiation
    # gaussian_tensor2 = ne.evaluate('exp(gaussian_tensor)')
    # gaussian_tensor2 *=(1. / (eta * np.sqrt(2*np.pi))) 
    # t1 = time.time()
    # print("exp", t1-t0, "secs")


    # assert np.allclose(gaussian_tensor, gaussian_tensor2)

    # t0 = time.time()
    summ = np.einsum("m..., im -> i...", weights, gaussian_tensor)
    # t1 = time.time()
    # print("sum", t1-t0, "secs")

    return summ
