# -*- coding: utf-8 -*-
"""
Basic routines for an ensemble

:author: Alexander Wietek
"""
import numpy as np
import scipy as sp
import scipy.linalg
from scipy.special import binom
from collections import OrderedDict
import time
import multiprocessing
import functools
from joblib import Parallel, delayed

import pytpq.statistics_for_tpq as st
import pytpq.linalg as pla
import pytpq.basic as pba


import matplotlib.pyplot as plt

def qn_moment_sum(ensemble,  qn_to_val, temperatures,shifts=None, k=1, 
                  e0=None, alpha_tag="AlphasV", beta_tag="BetasV", 
                  crop=True, check_posdef=True, ncores=None,
                  maxdepth=None):
    print("Called qn_moment_sum")
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
    temperatures = np.array(temperatures)
    betas = 1. / temperatures    

    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)

    qn_sum = OrderedDict()
    for seed in ensemble.seeds:
        summ = 0
        for qn in ensemble.qns:
            degeneracy = ensemble.degeneracy[qn]
            dimension = ensemble.dimension[qn]
            if degeneracy != 0 and dimension > 0:
                diag, offdiag = pba.tmatrix(ensemble, seed, qn, alpha_tag, 
                                            beta_tag, crop, maxdepth=maxdepth)
                #print("seed", seed, "qn", qn)
                #print("k",  k)
                #print("qn_to_val", qn_to_val(qn))
                #print("qn_to_val**k", qn_to_val(qn)**k)
                if shifts == None:
                    moment_avg = pla.moment_average(diag, offdiag, e0[seed], 
                                                    betas, None, 0, check_posdef)
                else:
                    moment_avg = pla.moment_average(diag, offdiag, e0[seed],
                                                    betas, shifts[seed][qn], 
                                                    0, check_posdef)
                #print("moment_avg", moment_avg)
                #print("degerneracy", degeneracy)
                #print("dimension", dimension, binom(16, int(qn[0]))*binom(16, int(qn[1])))
                #print()
                summ += qn_to_val(qn)**k * moment_avg * degeneracy * dimension
                # from scipy.special import binom
                # print(qn, qn_to_val(qn)**k, dimension, degeneracy, binom(16, int(qn[0]))*binom(16, int(qn[1])))
        qn_sum[seed] = summ
    return qn_sum


def moment_sum(ensemble, temperatures, shifts=None, k=0, e0=None,  
               alpha_tag="AlphasV", beta_tag="BetasV", 
               crop=True, check_posdef=True, ncores=None,
               maxdepth=None):
    print("Called moment_sum")
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

def operator_sum(ensemble, temperatures, operator_tag, 
                 shifts=None, k=1, e0=None,  
                 alpha_tag="AlphasV", beta_tag="BetasV", 
                 crop=True, check_posdef=True, ncores=None,
                 maxdepth=None):
    print("Called operator_sum")
    """ Get the sum of moments of the trigdiagonal matrices. 

    An operator sum average for a given quantum number sector is defined by

    (A)_{ij}^{(k)} = e_0 \exp( - \beta_i/2 (T - e0 + \mu_j) A \exp( - \beta_i/2 (T - e0 + \mu_j) e_0
    
    where \beta denotes inverse temperatures, \mu energy shifts for 
    a quantum number sector, and T the tridiagonal matrix, deg the
    degeneracy of the quantum number, and dim the dimension of the
    quantum number sector

    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        shifts         : optional, dictionary of shifts for every seed and qn
        k              : moment of operator (default: 1)
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
    temperatures = np.array(temperatures)
    betas = 1. / temperatures    

    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)
    op_sum = OrderedDict()
    for seed in ensemble.seeds:
        summ = 0
        for qn in ensemble.qns:
            # print("op, seed", seed)
            # print("op, qn", qn)
            degeneracy = ensemble.degeneracy[qn]
            dimension = ensemble.dimension[qn]
            if degeneracy != 0 and dimension > 0:
                diag, offdiag = pba.tmatrix(ensemble, seed, qn, alpha_tag, 
                                            beta_tag, crop, maxdepth=maxdepth)

                operator = ensemble.data(seed, qn, operator_tag)

                # print(operator)

                # Resize operator if tmatrix has been cropped (when beta small)
                size = len(diag)
                size_nocrop = len(ensemble.data(seed, qn, alpha_tag))
                if size_nocrop > size:
                    operator = operator[:size, :size]

                if operator.shape != (size, size):
                    raise ValueError("Incompatible shape {} of operator for tmatrix of length {}".format(
                        operator.shape, diag.shape))

                if shifts == None:
                    avg = pla.operator_average(diag, offdiag, operator, e0[seed], 
                                               betas, None, k,check_posdef)
                    # part = pla.moment_average(diag, offdiag, e0[seed], 
                    #                                 betas, None, 0, check_posdef, maxdepth)
                    # print("qn", qn)
                    # print("avg", avg[:3])
                    # print("part", part[:3])
                    # print("avg/part", (avg / part)[:3])
                else:
                    avg = pla.operator_average(diag, offdiag, operator, e0[seed],
                                               betas, shifts[seed][qn], 
                                               k, check_posdef)


 
                    
                summ += avg * degeneracy * dimension
                # print()
                # print()
        op_sum[seed] = summ
        # print("SUMM", summ)
    return op_sum




def _moment_sum_seed(seed, ensemble, temperatures, shifts=None, k=0, e0=None,  
                     alpha_tag="AlphasV", beta_tag="BetasV", 
                     crop=True, check_posdef=True,
                     maxdepth=None):
    # print(seed)
    print("Called _moment_sum_seed for seed", seed)
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



def partition(ensemble, temperatures, shifts=None, e0=None,  
              alpha_tag="AlphasV", beta_tag="BetasV", 
              crop=True, check_posdef=True, ncores=None,
              maxdepth=None):
    print("Called partition")
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
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)

    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    return st.mean(Z), st.error(Z)

        
def energy(ensemble, temperatures, shifts=None, e0=None,  
           alpha_tag="AlphasV", beta_tag="BetasV", 
           crop=True, check_posdef=True, ncores=None,
           maxdepth=None):
    print("Called energy")
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
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag)

    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    E = moment_sum(ensemble, temperatures, shifts, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)

    energy = OrderedDict()
    for seed in ensemble.seeds:
        energy[seed] = E_jackknife[seed] / Z_jackknife[seed]

    return st.mean(energy), st.error_jackknife(energy)


def entropy(ensemble, temperatures, shifts=None, e0=None,  
            alpha_tag="AlphasV", beta_tag="BetasV", 
            crop=True, check_posdef=True, ncores=None,
            maxdepth=None):
    print("Called entropy")
    """ Get the entropy for a given set of temperatures and shifts
    
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
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)


    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    E = moment_sum(ensemble, temperatures, shifts, 1, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)

    entropy = OrderedDict()
    for seed in ensemble.seeds:
        energy_term = E_jackknife[seed] / Z_jackknife[seed] - e0[seed]
        # divide by temperatures
        if shifts is None:
            energy_term /= temperatures
        else:
            for idx in range(energy_term.shape[1]):
                energy_term[:,idx] /= temperatures
        entropy[seed] = np.log(Z_jackknife[seed]) + energy_term
                        
    return st.mean(entropy), st.error_jackknife(entropy)


def specific_heat(ensemble, temperatures, shifts=None, e0=None,  
                  alpha_tag="AlphasV", beta_tag="BetasV", 
                  crop=True, check_posdef=True, ncores=None,
                  maxdepth=None):
    print("Called specific_heat")
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
    temperatures = np.array(temperatures)

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
                   crop=True, check_posdef=True, ncores=None,
                   maxdepth=None):
    print("Called thermodynamics")
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


def quantumnumber(ensemble, qn_to_val, temperatures, shifts=None, e0=None,  
                  alpha_tag="AlphasV", beta_tag="BetasV", 
                  crop=True, check_posdef=True, ncores=None,
                  maxdepth=None):
    print("Called quantumnumber")
    """ Get quantum number average for a given set of temperatures and shifts
    
    Args:
        ensemble       : Ensemble class
        qn_to_val      : function translating a quantum number to float
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
    temperatures = np.array(temperatures)
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)

    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores,
                   maxdepth)
    N = qn_moment_sum(ensemble,  qn_to_val, temperatures, shifts, 1, 
                      e0, alpha_tag, beta_tag, crop, check_posdef, ncores,
                      maxdepth)

    Z_jackknife = st.jackknife(Z)
    N_jackknife = st.jackknife(N)

    QN = OrderedDict()
    for seed in ensemble.seeds:
        QN[seed] = N_jackknife[seed] / Z_jackknife[seed]

    return st.mean(QN), st.error_jackknife(QN)


def susceptibility(ensemble, qn_to_val, temperatures, shifts=None, e0=None,  
                   alpha_tag="AlphasV", beta_tag="BetasV", 
                   crop=True, check_posdef=True, ncores=None,
                   maxdepth=None):
    print("Called susceptibility")
    """ Get quantum number susceptibility for given set of tempraturs and shifts
    
    Args:
        ensemble       : Ensemble class
        qn_to_val      : function translating a quantum number to float
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
    temperatures = np.array(temperatures)
    if e0 == None:
        e0, e0qns = pba.ground_state_energy(ensemble, shifts=shifts, ncores=ncores, 
                                            alpha_tag=alpha_tag, beta_tag=beta_tag,
                                            maxdepth=maxdepth)

    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    N = qn_moment_sum(ensemble,  qn_to_val, temperatures, shifts, 1, 
                      e0, alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)
    N2 = qn_moment_sum(ensemble,  qn_to_val, temperatures, shifts, 2, 
                      e0, alpha_tag, beta_tag, crop, check_posdef, ncores, maxdepth)

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


def operator(ensemble, temperatures, operator_tag, shifts=None, e0=None,  
             alpha_tag="AlphasV", beta_tag="BetasV", 
             crop=True, check_posdef=True, ncores=None,
             maxdepth=None):
    print("Called operator")
    """ Get quantum number susceptibility for given set of tempraturs and shifts
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        operator_tag   : string, tag that defines the operator matrix in Lanczos basis
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
    temperatures = np.array(temperatures)

    Z = moment_sum(ensemble, temperatures, shifts, 0, e0,  
                   alpha_tag, beta_tag, crop, check_posdef, ncores)
    O = operator_sum(ensemble, temperatures, operator_tag, shifts, 1, 
                     e0, alpha_tag, beta_tag, crop, check_posdef, ncores)
 
    Z_jackknife = st.jackknife(Z)
    O_jackknife = st.jackknife(O)

    op = OrderedDict()
    for seed in ensemble.seeds:
        op[seed] = O_jackknife[seed] / Z_jackknife[seed]

    return st.mean(op), st.error_jackknife(op)
