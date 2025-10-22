# -*- coding: utf-8 -*-
"""
Basic routines for an ensemble

:author: Alexander Wietek
"""
import numpy as np
from collections import OrderedDict
import time
import multiprocessing
import functools

import pytpq.statistics_for_tpq as st
import pytpq.linalg as pla
import pytpq.basic as pba

def moment_sum(ensemble, temperatures, k=0, e0=None,  
               alpha_tag="Alphas", beta_tag="Betas", 
               crop=True, check_posdef=True,
               maxdepth=None):
    """ Get the sum of moments of the trigdiagonal matrices. 

    A moment average for a given quantum number sector is defined by

    M_{ij}^{(k)}= e_0 T^k \exp( - \beta_i (T - e0 + \mu_j) e_0 * deg * dim
    
    where \beta denotes inverse temperatures, and T the tridiagonal matrix, deg the
    degeneracy of the quantum number, and dim the dimension of the
    quantum number sector

    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        k              : moment of eigenvalues
        e0             : precomputed ground state energy, optional
        alpha_tag : string, which tag is chosen for diagonal data
        beta_tag  : string, which tag is chosen for offdiagonal data
        crop      : flag whether tmatrix is cropped on deflation (def: True)
        check_posdef : check whether all weights are positive in exponentiation
    Returns:
        OrderedDict : sum of moment averages for a given seed
    """

    if not np.all(temperatures) > 0:
        raise ValueError("Invalid temperatures")
    
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)
    moment_sum = OrderedDict()
    for seed in ensemble.seeds:
        _, summ = _moment_sum_seed(seed, ensemble, temperatures, k, e0, alpha_tag, 
                                    beta_tag, crop, check_posdef,
                                    maxdepth=maxdepth)
        moment_sum[seed] = summ

    return moment_sum


def _moment_sum_seed(seed, ensemble, temperatures, k=0, e0=None,  
                     alpha_tag="Alphas", beta_tag="Betas", 
                     crop=True, check_posdef=True,
                     maxdepth=None):
    # print(seed)
    betas = 1. / temperatures

    summ = 0
    for qn in ensemble.qns:
        degeneracy = ensemble.degeneracy[qn]
        dimension = ensemble.dimension[qn]
        if degeneracy != 0 and dimension > 0:
            diag, offdiag = pba.tmatrix(ensemble, seed, qn, alpha_tag, beta_tag, crop, maxdepth=maxdepth)
            moment_avg = pla.moment_average(diag, offdiag, e0[seed],  betas, k, check_posdef)
            summ += moment_avg * degeneracy * dimension

    return seed, summ


"""
    DEBUG ONLY:
    Compute moment estimates at temperature beta separately for each qn sector but average over seeds.
    NO DEGENRACIES INCLUDED!!!
"""
def moment_sector_check(ensemble, beta, e0, k=0,
                        alpha_tag="Alphas", beta_tag="Betas", 
                        crop=True, check_posdef=True,
                        maxdepth=None):
    
    moment_averages = OrderedDict()
    for qn in ensemble.qns:
        moment_avg_qn = 0.0
        for seed in ensemble.seeds:
            diag, offdiag = pba.tmatrix(ensemble, seed, qn, alpha_tag, beta_tag, crop, maxdepth=maxdepth)
            moment_avg = pla.moment_average(diag, offdiag, e0[seed],  np.array([beta]), k, check_posdef)
            moment_avg_qn += moment_avg[0] * ensemble.dimension[qn]
        moment_averages[qn] = moment_avg_qn / len(ensemble.seeds)

    return moment_averages


"""
    Compute thermodynamic quantities on each qn sector separately and a high temperature
    but average over all seeds (to check against full ED).
"""
def high_T_sector_check(ensemble, temperature,  
                        alpha_tag="Alphas", beta_tag="Betas", 
                        crop=True, check_posdef=True,
                        maxdepth=None):
    T = temperature[-1] # pick last (highest) temperature for the comparison
    beta = 1/T

    # use fixed energy offset for normalization in all sectors
    e0_val = -10
    e0 = {seed:e0_val for seed in ensemble.seeds} # dict sending all seeds to e0_val
    
    E_beta = moment_sector_check(ensemble, beta, e0, k=1,
                                  alpha_tag=alpha_tag, beta_tag=beta_tag,
                                  crop=crop, check_posdef=check_posdef,
                                  maxdepth=maxdepth)
    
    # normalize energy by best ground state estimate
    for qn in ensemble.qns:
        E_beta[qn] = E_beta[qn] / abs(e0_val)
    
    return E_beta



def thermodynamics(ensemble, temperatures, e0=None,  
                   alpha_tag="Alphas", beta_tag="Betas", 
                   crop=True, check_posdef=True,
                   maxdepth=None):
    """ Get the partition / energy / specific heat 
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        k              : moment of eigenvalues
        e0             : precomputed ground state energy, optional
        alpha_tag      : string, which tag is chosen for diagonal data
        beta_tag       : string, which tag is chosen for offdiagonal data
        crop           : flag whether tmatrix is cropped on deflation (def: True)
        check_posdef   : check whether all weights are positive in exponentiation
    Returns:
        np.array, np.array: mean / error estimate for specific heat for 
                            every temperature
    """
    temperatures = np.array(temperatures)

    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)

    Z = moment_sum(ensemble, temperatures, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, maxdepth)
    E = moment_sum(ensemble, temperatures, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, maxdepth)
    Q = moment_sum(ensemble, temperatures, 2, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, maxdepth)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)
    Q_jackknife = st.jackknife(Q)

    energy = OrderedDict()
    for seed in ensemble.seeds:
        energy[seed] = E_jackknife[seed] / Z_jackknife[seed]
    
    specific_heat = OrderedDict()
    betas = 1. / temperatures
    for seed in ensemble.seeds:
        specific_heat[seed] = ( Q_jackknife[seed] / Z_jackknife[seed] - (E_jackknife[seed] / Z_jackknife[seed])**2)
        specific_heat[seed] = betas**2 * specific_heat[seed]

    return st.mean(Z), st.error(Z), \
        st.mean(energy), st.error_jackknife(energy), \
        st.mean(specific_heat), st.error_jackknife(specific_heat)


def entropy(ensemble, temperatures, e0=None,  
            alpha_tag="AlphasV", beta_tag="BetasV", 
            crop=True, check_posdef=True, ncores=None,
            maxdepth=None):
    """ Get the entropy for a given set of temperatures and shifts
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        k              : moment of eigenvalues
        e0             : precomputed ground state energy, optional
        alpha_tag      : string, which tag is chosen for diagonal data
        beta_tag       : string, which tag is chosen for offdiagonal data
        crop           : flag whethr tmatrix is cropped on deflation (def: True)
        check_posdef   : check whethr all weights are positive in exponentiation
        ncores         : number of parallel processes used for the computation
    Returns:
        np.array, np.array: mean / error estimate for energy for 
                            every temperature and shift
    """
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)


    Z = moment_sum(ensemble, temperatures, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    E = moment_sum(ensemble, temperatures, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)

    entropy = OrderedDict()
    for seed in ensemble.seeds:
        energy_term = E_jackknife[seed] / Z_jackknife[seed] - e0[seed]
        # divide by temperatures
        energy_term /= temperatures
        entropy[seed] = np.log(Z_jackknife[seed]) + energy_term
                        
    return st.mean(entropy), st.error_jackknife(entropy)

