# -*- coding: utf-8 -*-
"""
Basic routines for an ensemble

:author: Alexander Wietek
"""
import numpy as np
import scipy as sp
import scipy.linalg
from collections import OrderedDict
import time
import multiprocessing
import functools
from joblib import Parallel, delayed

import pytpq.statistics_for_tpq as st
import pytpq.linalg as pla


def tmatrix(ensemble, seed, qn, alpha_tag="AlphasV", beta_tag="BetasV",
            crop=True):
    """ Return (cropped) Tmatrix 

    Args:
        ensemble  : Ensemble class
        seed      : random seed
        qn        : quantum number
        alpha_tag : string, which tag is chosen for diagonal data
        beta_tag  : string, which tag is chosen for offdiagonal data
        crop      : flag whether tmatrix is cropped on deflation (def: True)
    Returns:
        np.array, np.array: diagonal and offdiagonal part of Tmatrix
    """
    diag = ensemble.data(seed, qn, alpha_tag)
    offdiag = ensemble.data(seed, qn, beta_tag)[:-1]

    # remove overiterated tmatrix entries
    if crop and np.any(np.abs(offdiag) < 1e-10):
        diag = diag[:(np.argmax(np.abs(offdiag) < 1e-10)+1)]
        offdiag = offdiag[:np.argmax(np.abs(offdiag) < 1e-10)]

    return diag, offdiag


def get_shifts(ensemble, multipliers, qn_to_val):
    """ Compute shifts in energy for a given modifier and multipliers 

    Args:
        ensemble       : Ensemble class
        multiplier     : array of Lagrange multipliers (e.g magnetic field)
        qn_to_val      : function translating a quantum number to float
    Returns:
        dict :  dictionaries of dictionaries with shifts for seed and qn
    """
    shifts = dict()
    for seed in ensemble.seeds:
        shifts[seed] = dict()
        for qn in ensemble.qns:
            shifts[seed][qn] = multipliers * qn_to_val(qn)
    return shifts


def ground_state_energy(ensemble, alpha_tag="AlphasV", beta_tag="BetasV", 
                        shifts=None, ncores=None):
    """ Get the total ground state energy for all seeds of an ensemble
    
    Args:
        ensemble  : Ensemble class
        alpha_tag : string, which tag is chosen for alpha data
        beta_tag  : string, which tag is chosen for beta data
        shifts    : optional, dictionary of shifts for every seed and qn
        ncores    : number of parallel processes used for the computation
    Returns:
        OrderedDict, OrderedDict:  dictionaries of ground state energies, 
                                   and corresponding quantum number sector
    """
    e0s = OrderedDict()
    e0_qns = OrderedDict()

    # Get ground state energy for every seed serial
    if ncores == None:
        for seed in ensemble.seeds:
            _, e0, e0_qn = _ground_state_energy_seed(seed, ensemble, 
                                                        alpha_tag="AlphasV", 
                                                        beta_tag="BetasV", 
                                                        shifts=shifts)
            e0s[seed] = e0
            e0_qns[seed] = e0_qn

    # Parallelization over seeds
    else:
        e0_func = functools.partial(_ground_state_energy_seed,
                                    ensemble=ensemble, alpha_tag=alpha_tag,
                                    beta_tag=beta_tag, shifts=shifts)
        # with multiprocessing.Pool(ncores) as p:
        #     seeds = ensemble.seeds
        #     results = p.map(e0_func, seeds)

        results = Parallel(n_jobs=ncores, backend="threading")\
                  (map(delayed(e0_func), ensemble.seeds))


        for seed, e0, e0_qn in results:
            e0s[seed] = e0
            e0_qns[seed] = e0_qn

    return e0s, e0_qns



def _ground_state_energy_seed(seed, ensemble, alpha_tag="AlphasV", 
                              beta_tag="BetasV", shifts=None):
    """ Get the total ground state energy for a seed of an ensemble """
    assert len(ensemble.qns) > 0

    # Prepare data structure holding e0 and its qn
    if shifts == None:
        e0 = np.inf
        e0_qn = ""
    else:
        n_shifts = len(shifts[seed][ensemble.qns[0]])
        e0 = np.full(n_shifts, np.inf)
        e0_qn = np.array([ensemble.qns[0]] * n_shifts)

    # find energy minimizing quantum number sector
    for qn in ensemble.qns:
        if ensemble.degeneracy[qn] > 0 and ensemble.dimension[qn] > 0:
            diag, offdiag = tmatrix(ensemble, seed, qn, alpha_tag, 
                                    beta_tag, crop=True)
            if shifts == None:
                e = pla.tmatrix_e0(diag, offdiag)
                if e < e0:
                    e0 = e
                    e0_qn = qn
            else:
                es = pla.tmatrix_e0(diag, offdiag) + shifts[seed][qn]
                smaller_indices = es < e0
                e0[smaller_indices] = es[smaller_indices]
                e0_qn[smaller_indices] = qn

    t1 = time.time()
    return seed, e0, e0_qn


def qn_moment_sum(ensemble,  qn_to_val, temperatures,shifts=None, k=1, 
                  e0=None, alpha_tag="AlphasV", beta_tag="BetasV", 
                  crop=True, check_posdef=True, ncores=None):
    """ Get the sum of values derived from quantum numbers 

    A quantum number average for a given quantum number sector is defined by

    Q_{ij}^{(k)}= e_0 qn_val \exp( - \beta_i (T - e0 + \mu_j) e_0 * deg * dim
    
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

    betas = 1. / temperatures    

    if e0 == None:
        e0, _ = ground_state_energy(ensemble, shifts=shifts, ncores=ncores)

    qn_sum = OrderedDict()
    for seed in ensemble.seeds:
        print(seed)
        summ = 0
        for qn in ensemble.qns:
            degeneracy = ensemble.degeneracy[qn]
            dimension = ensemble.dimension[qn]
            if degeneracy != 0 and dimension > 0:
                diag, offdiag = tmatrix(ensemble, seed, qn, alpha_tag, 
                                        beta_tag, crop)
                if shifts == None:
                    moment_avg = pla.moment_average(diag, offdiag, e0[seed], 
                                                    betas, None, 0,check_posdef)
                else:
                    moment_avg = pla.moment_average(diag, offdiag, e0[seed],
                                                    betas, shifts[seed][qn], 
                                                    0, check_posdef)
                summ += qn_to_val(qn)**k * moment_avg * degeneracy * dimension
        qn_sum[seed] = summ
    return qn_sum


def moment_sum(ensemble, temperatures, shifts=None, k=0, e0=None,  
               alpha_tag="AlphasV", beta_tag="BetasV", 
               crop=True, check_posdef=True, ncores=None):
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
        e0, _ = ground_state_energy(ensemble, shifts=shifts, ncores=ncores)

    moment_sum = OrderedDict()
    if ncores == None:
        for seed in ensemble.seeds:
            _, summ = _moment_sum_seed(seed, ensemble, temperatures, 
                                       shifts, k, e0, alpha_tag, 
                                       beta_tag, crop, check_posdef)
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
                     crop=True, check_posdef=True):
    print(seed)

    betas = 1. / temperatures

    summ = 0
    for qn in ensemble.qns:
        degeneracy = ensemble.degeneracy[qn]
        dimension = ensemble.dimension[qn]
        if degeneracy != 0 and dimension > 0:
            diag, offdiag = tmatrix(ensemble, seed, qn, alpha_tag, 
                                    beta_tag, crop)
            if shifts == None:
                moment_avg = pla.moment_average(diag, offdiag, e0[seed], 
                                                betas, None, k,check_posdef)
            else:
                moment_avg = pla.moment_average(diag, offdiag, e0[seed],
                                                betas, shifts[seed][qn], k, 
                                                check_posdef)
            summ += moment_avg * degeneracy * dimension

    return seed, summ



def partition(ensemble, temperatures, shifts=None, e0=None,  
              alpha_tag="AlphasV", beta_tag="BetasV", 
              crop=True, check_posdef=True, ncores=None):
    """ Get the partition function for a given set of temperatures and shifts
    
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
        np.array, np.array: mean / error estimate for partition function for 
                            every temperature and shift
    """
    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    return st.mean(Z), st.error(Z)

        
def energy(ensemble, temperatures, shifts=None, e0=None,  
           alpha_tag="AlphasV", beta_tag="BetasV", 
           crop=True, check_posdef=True, ncores=None):
    """ Get the energy for a given set of temperatures and shifts
    
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
        np.array, np.array: mean / error estimate for energy for 
                            every temperature and shift
    """
    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    E = moment_sum(ensemble, temperatures, shifts, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)

    energy = OrderedDict()
    for seed in ensemble.seeds:
        energy[seed] = E_jackknife[seed] / Z_jackknife[seed]

    return st.mean(energy), st.error_jackknife(energy)


def specific_heat(ensemble, temperatures, shifts=None, e0=None,  
                  alpha_tag="AlphasV", beta_tag="BetasV", 
                  crop=True, check_posdef=True, ncores=None):
    """ Get the specific heat for a given set of temperatures and shifts
    
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
    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    E = moment_sum(ensemble, temperatures, shifts, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    Q = moment_sum(ensemble, temperatures, shifts, 2, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)
    Q_jackknife = st.jackknife(Q)
    

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

    return st.mean(specific_heat), st.error_jackknife(specific_heat)


def thermodynamics(ensemble, temperatures, shifts=None, e0=None,  
                   alpha_tag="AlphasV", beta_tag="BetasV", 
                   crop=True, check_posdef=True, ncores=None):
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
    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    E = moment_sum(ensemble, temperatures, shifts, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    Q = moment_sum(ensemble, temperatures, shifts, 2, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)

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


def quantumnumber(ensemble, qn_to_val, temperatures, shifts=None, e0=None,  
                  alpha_tag="AlphasV", beta_tag="BetasV", 
                  crop=True, check_posdef=True, ncores=None):
    """ Get quantum number average for a given set of temperatures and shifts
    
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
        np.array, np.array: mean / error estimate for partition function for 
                            every temperature and shift
    """
    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    N = qn_moment_sum(ensemble,  qn_to_val, temperatures, shifts, 1, 
                      e0, alpha_tag, beta_tag, crop, check_posdef, ncores)

    Z_jackknife = st.jackknife(Z)
    N_jackknife = st.jackknife(N)

    QN = OrderedDict()
    for seed in ensemble.seeds:
        QN[seed] = N_jackknife[seed] / Z_jackknife[seed]

    return st.mean(QN), st.error_jackknife(QN)


def susceptibility(ensemble, qn_to_val, temperatures, shifts=None, e0=None,  
                   alpha_tag="AlphasV", beta_tag="BetasV", 
                   crop=True, check_posdef=True, ncores=None):
    """ Get quantum number susceptibility for given set of tempraturs and shifts
    
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
        np.array, np.array: mean / error estimate for partition function for 
                            every temperature and shift
    """
    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    N = qn_moment_sum(ensemble,  qn_to_val, temperatures, shifts, 1, 
                      e0, alpha_tag, beta_tag, crop, check_posdef, ncores)
    N2 = qn_moment_sum(ensemble,  qn_to_val, temperatures, shifts, 2, 
                      e0, alpha_tag, beta_tag, crop, check_posdef, ncores)

    Z_jackknife = st.jackknife(Z)
    N_jackknife = st.jackknife(N)
    N2_jackknife = st.jackknife(N2)

    susceptibility = OrderedDict()
    betas = 1. / temperatures
    for seed in ensemble.seeds:
        susceptibility[seed] = ( N2_jackknife[seed] / Z_jackknife[seed] \
                                - (N_jackknife[seed] / Z_jackknife[seed])**2)

        # Multiply by betas
        if shifts is None:
            susceptibility[seed] = betas * susceptibility[seed]
        else:
            susceptibility[seed] = np.einsum("i, ij->ij", 
                                            betas, susceptibility[seed])


    return st.mean(susceptibility), st.error_jackknife(susceptibility)
