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
from joblib import Parallel, delayed

import pytpq.statistics_for_tpq as st
import pytpq.linalg as pla
import pytpq.basic as pba

def moment_sum(ensemble, temperatures, shifts=None, k=0, e0=None,  
               alpha_tag="AlphasV", beta_tag="BetasV", 
               crop=True, check_posdef=True, ncores=None,
               maxdepth=None):
    """ Get the sum of moments of the trigdiagonal matrices. 

    A moment average for a given quantum number sector is defined by

    M_{ij}^{(k)}= e_0 T^k \exp( - \beta_i (T - e0 + \mu_j) e_0 * deg * dim
    
    where \beta denotes inverse temperatures, \mu energy shifts for 
    a quantum number sector, and T the tridiagonal matrix, deg the
    degeneracy of the quantum number, and dim the dimension of the
    quantum number sector

    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        shifts         : optional, dictionary of shifts for every seed and qn
        k              : moment of eigenvalues
        e0             : precomputed ground state energy, optional
        alpha_tag : string, which tag is chosen for diagonal data
        beta_tag  : string, which tag is chosen for offdiagonal data
        crop      : flag whether tmatrix is cropped on deflation (def: True)
        check_posdef : check whether all weights are positive in exponentiation
        ncores    : number of parallel processes used for the computation
    Returns:
        OrderedDict : sum of moment averages for a given seed
    """

    if not np.all(temperatures) > 0:
        raise ValueError("Invalid temperatures")
    
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)
    moment_sum = OrderedDict()
    if ncores == None:
        for seed in ensemble.seeds:
            _, summ = _moment_sum_seed(seed, ensemble, temperatures, 
                                       shifts, k, e0, alpha_tag, 
                                       beta_tag, crop, check_posdef,
                                       maxdepth=maxdepth)
            moment_sum[seed] = summ

    # Parallelization over seeds
    else:
        sum_func = functools.partial(_moment_sum_seed, ensemble=ensemble, 
                                     temperatures=temperatures, shifts=shifts, 
                                     k=k, e0=e0, alpha_tag=alpha_tag,
                                     beta_tag=beta_tag, crop=crop, 
                                     check_posdef=check_posdef)
        with multiprocessing.Pool(ncores) as p:
            results = p.map(sum_func, ensemble.seeds)

        for seed, summ in results:
            moment_sum[seed] = summ

    return moment_sum


def _moment_sum_seed(seed, ensemble, temperatures, shifts=None, k=0, e0=None,  
                     alpha_tag="AlphasV", beta_tag="BetasV", 
                     crop=True, check_posdef=True,
                     maxdepth=None):
    # print(seed)
    betas = 1. / temperatures

    summ = 0
    for qn in ensemble.qns:
        # print(qn)
        degeneracy = ensemble.degeneracy[qn]
        dimension = ensemble.dimension[qn]
        if degeneracy != 0 and dimension > 0:
            diag, offdiag = pba.tmatrix(ensemble, seed, qn, alpha_tag, 
                                        beta_tag, crop, maxdepth=maxdepth)
            if shifts == None:
                moment_avg = pla.moment_average(diag, offdiag, e0[seed], 
                                                betas, None, k, check_posdef)
            else:
                moment_avg = pla.moment_average(diag, offdiag, e0[seed],
                                                betas, shifts[seed][qn], k, 
                                                check_posdef)
            # I have no clue why the lines below are here, so I uncommented them
            #from scipy.special import binom
            #print(qn, dimension, degeneracy, binom(16, int(qn[0]))*binom(16, int(qn[1])))
            #print("avg", moment_avg * degeneracy * dimension)
            #print()
            summ += moment_avg * degeneracy * dimension

    return seed, summ


def thermodynamics(ensemble, temperatures, shifts=None, e0=None,  
                   alpha_tag="AlphasV", beta_tag="BetasV", 
                   crop=True, check_posdef=True, ncores=None,
                   maxdepth=None):
    """ Get the partition / energy / specific heat for a given set 
        of temperatures and shifts
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        shifts         : optional, dictionary of shifts for every seed and qn
        k              : moment of eigenvalues
        e0             : precomputed ground state energy, optional
        alpha_tag      : string, which tag is chosen for diagonal data
        beta_tag       : string, which tag is chosen for offdiagonal data
        crop           : flag whethr tmatrix is cropped on deflation (def: True)
        check_posdef   : check whethr all weights are positive in exponentiation
        ncores         : number of parallel processes used for the computation
    Returns:
        np.array, np.array: mean / error estimate for specific heat for 
                            every temperature and shift
    """
    temperatures = np.array(temperatures)

    for qn in ensemble.qns:
        degeneracy = ensemble.degeneracy[qn]
        print("Quantum number:", qn, "has degeneracy", degeneracy)

    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)

    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    E = moment_sum(ensemble, temperatures, shifts, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    Q = moment_sum(ensemble, temperatures, shifts, 2, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)
    Q_jackknife = st.jackknife(Q)

    energy = OrderedDict()
    for seed in ensemble.seeds:
        energy[seed] = E_jackknife[seed] / Z_jackknife[seed]
    
    specific_heat = OrderedDict()
    betas = 1. / temperatures
    for seed in ensemble.seeds:
        specific_heat[seed] = ( Q_jackknife[seed] / Z_jackknife[seed] \
                                - (E_jackknife[seed] / Z_jackknife[seed])**2)

        # Multiply by betas
        if shifts is None:
            specific_heat[seed] = betas**2 * specific_heat[seed]
        else:
            specific_heat[seed] = np.einsum("i, ij->ij", 
                                            betas**2, specific_heat[seed])

    return st.mean(Z), st.error(Z), \
        st.mean(energy), st.error_jackknife(energy), \
        st.mean(specific_heat), st.error_jackknife(specific_heat)
