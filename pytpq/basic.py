# -*- coding: utf-8 -*-
"""
Basic routines for an ensemble

:author: Alexander Wietek
"""
import numpy as np
from collections import OrderedDict
import time
import multiprocessing

import  pytpq.statistics_for_tpq as st


def ground_state_energy(ensemble, tag="EigenvaluesV", modifier=None):
    """ Get the total ground state energy for every seed of an ensemble
    
    Args:
        ensemble       : Ensemble class
        tag            : string, which tag is chosen for eigenvalue data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
    Returns:
        OrderedDict, OrderedDict:  dictionaries of ground state energies, 
                                   and corresponding quantum number sector
    """
    e0s = OrderedDict()
    e0s_qns = OrderedDict()

    # Get ground state energy for every sector
    for seed in ensemble.seeds:
        e0s_for_qns = []

        for qn in ensemble.qns:
            if ensemble.degeneracy[qn] > 0 and ensemble.dimension[qn] > 0:
                if modifier != None:
                    e0s_for_qns.append((min(modifier(qn, ensemble.data(seed, qn, tag))), qn))
                else:
                    e0s_for_qns.append((min(ensemble.data(seed, qn, tag)), qn))

        e0_qn = min(e0s_for_qns, key=lambda e0_qn: e0_qn[0])    
        e0s[seed] = e0_qn[0]
        e0s_qns[seed] = e0_qn[1]
        
    return e0s, e0s_qns



def get_sum_of_function(ensemble, tpq_function, eigenvalue_function,
                        alpha_tag="AlphasV", beta_tag="BetasV", 
                        eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get a sum of a function applied to the eigenvalues
    
    Args:
        ensemble       : Ensemble class
        function       : function of seed, quantum number and np.array of values to sum up
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """

    # Compute sum for all seeds
    sums = OrderedDict()

    def evaluate_single_seed(seed):
        print("seed", seed)

        # Accumulate all quantum number sectors
        first = True
        for qn in ensemble.qns:

            degeneracy = ensemble.degeneracy[qn]
            if degeneracy != 0 and ensemble.dimension[qn] > 0:

                # Use exact function if possible
                if ensemble.exact[qn]:
                    if modifier != None:
                        fun_evaluated = eigenvalue_function(seed, qn, \
                            modifier(qn, ensemble.data(seed, qn, eigenvalue_tag)))
                    else:
                        fun_evaluated = eigenvalue_function(seed, qn, \
                            ensemble.data(seed, qn, eigenvalue_tag))
 
                else:
                    diag = ensemble.data(seed, qn, alpha_tag)
                    offdiag = ensemble.data(seed, qn, beta_tag)[:-1]

                    # remove overiterated tmatrix entries
                    if np.any(np.abs(offdiag) < 1e-10):
                        diag = diag[:(np.argmax(np.abs(offdiag) < 1e-10)+1)]
                        offdiag = offdiag[:np.argmax(np.abs(offdiag) < 1e-10)]

                    fun_evaluated = tpq_function(seed, qn, \
                        diag, offdiag, modifier)
            
                if first:
                    sums[seed] = fun_evaluated * degeneracy
                    first = False
                else:
                    sums[seed] += fun_evaluated * degeneracy

    for seed in ensemble.seeds:
        evaluate_single_seed(seed)
    
    return sums


def exp_betas_outer_tmatrix(betas, tmatrix):
    # t0 = time.time()
    teigs, tvecs = np.linalg.eigh(tmatrix)

    # tmatrix is assumed to be positive definite
    if not np.all(teigs > -1e-12):
        print(teigs)
    assert(np.all(teigs > -1e-12))

    # betas are assumed to be all negative
    assert(np.all(betas < 0))
    tvecs_herm = np.transpose(tvecs.conj())
    exp_betas_outer_teigs = np.exp(np.outer(betas, teigs))
    # t1 = time.time()
    # print("out ", t1-t0)

    
    # t0 = time.time()
    lst = np.empty((len(betas), tvecs.shape[0], tvecs.shape[1]))
    for idx, exp_teigs in enumerate(exp_betas_outer_teigs):
        lst[idx,:,:] = np.dot(tvecs, np.dot(np.diag(exp_teigs), tvecs_herm))


    # t1 = time.time()
    # print("lst ", t1-t0)

    # t0 = time.time()
    # lst2 = np.einsum('ij, tj, jl-> til', tvecs, exp_betas_outer_teigs, tvecs_herm)
    # t1 = time.time()
    # print("lst2", t1-t0)


    # print("close", max(lst - lst2))
    return lst



def moment(ensemble, temperatures, k, alpha_tag="AlphasV", beta_tag="BetasV",
           eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get moment of eigenvalues summed up (e.g. partition, energy, energy**2)
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        k              : moment of eigenvalues
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """

    if not np.all(temperatures > 0):
        raise ValueError("Invalid temperatures found!")
    betas = 1. / temperatures

    e0s, _ = ground_state_energy(ensemble, eigenvalue_tag, modifier)

    # #############################################
    # Function to evaluate moment with TPQ
    # Compute all  (exp(-beta/2 * (T - e0*Id)) * T**k * exp(-beta/2 * (T - e0*Id)))_00
    def tpq_function(seed, qn, diag, offdiag, modifier=None):

        tmatrix = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, - 1)

        # T + (mod - e0) *Id
        if modifier != None:
            expmats = exp_betas_outer_tmatrix(-betas/2, tmatrix + (-modifier(qn, 0) - e0s[seed])  * np.eye(len(diag)))
            tmat_power = np.linalg.matrix_power(tmatrix - modifier(qn, 0) * np.eye(len(diag)), k)
        else:
            expmats = exp_betas_outer_tmatrix(-betas/2, tmatrix - e0s[seed] * np.eye(len(diag)))
            tmat_power = np.linalg.matrix_power(tmatrix, k)

        funs = []
        for expmat in expmats:
            funs.append(np.dot(expmat[0,:], np.dot(tmat_power, expmat[:,0])))

        return np.array(funs) * ensemble.dimension[qn]


    # #############################################
    # Function to evaluate moment with exact eigenvalues
    # Compute \sum_i e_i^k * exp(-beta * (e_i - e0)) 
    def eigenvalue_function(seed, qn, eigenvalues):
        moment_matrix = np.outer(eigenvalues**k, np.ones_like(temperatures))
        expmatrix = np.multiply(moment_matrix, 
                                np.exp(-np.outer(eigenvalues - e0s[seed], betas)))
        return np.sum(expmatrix, axis=0)

    return get_sum_of_function(ensemble, tpq_function, eigenvalue_function,
                               alpha_tag, beta_tag, eigenvalue_tag, modifier)


def qnsum(ensemble, temperatures, qn_to_val, alpha_tag="AlphasV", beta_tag="BetasV",
          eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get sum of observable only dependent on the quantum number 
        (e.g. magnetization, particle number)
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        qn_to_val      : a function that transforms a qn to its float value
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """

    if not np.all(temperatures > 0):
        raise ValueError("Invalid temperatures found!")
    betas = 1. / temperatures

    e0s, _ = ground_state_energy(ensemble, eigenvalue_tag, modifier)

    # #############################################
    # Function to evaluate qn average with TPQ
    # Compute all  (exp(-beta* (T - e0*Id)) * qn)_00
    def tpq_function(seed, qn, diag, offdiag, modifier):

        tmatrix = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, - 1)
        
        # T + (mod - e0) *Id
        if modifier != None:
            expmats = exp_betas_outer_tmatrix(-betas/2, tmatrix + (-modifier(qn, 0) - e0s[seed])  * np.eye(len(diag)))
        else:
            expmats = exp_betas_outer_tmatrix(-betas/2, tmatrix - e0s[seed] * np.eye(len(diag)))

        
        return expmats[:,0,0] * qn_to_val(qn) * ensemble.dimension[qn] 

    # #############################################
    # Function to evaluate qn average with exact eigenvalues
    # Compute \sum_i qn * exp(-beta * (e_i - e0)) 
    def eigenvalue_function(seed, qn, eigenvalues):
        expmatrix = np.exp(-np.outer(eigenvalues - e0s[seed], betas))
        return np.sum(expmatrix, axis=0) * qn_to_val(qn)


    return get_sum_of_function(ensemble, tpq_function, eigenvalue_function,
                               alpha_tag, beta_tag, eigenvalue_tag, modifier)


def partition(ensemble, temperatures, alpha_tag="AlphasV", beta_tag="BetasV", 
              eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get the partition function for a given set of temperatures
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """
    Z = moment(ensemble, temperatures, 0, alpha_tag, beta_tag, eigenvalue_tag, modifier)
    
    return st.mean(Z), st.error(Z)

        
def energy(ensemble, temperatures, alpha_tag="AlphasV", beta_tag="BetasV", 
           eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get the partition function for a given set of temperatures
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """
    Z = moment(ensemble, temperatures, 0, alpha_tag, beta_tag, eigenvalue_tag, modifier)
    E = moment(ensemble, temperatures, 1, alpha_tag, beta_tag, eigenvalue_tag, modifier)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)

    energy = OrderedDict()
    for seed in ensemble.seeds:
        energy[seed] = E_jackknife[seed] / Z_jackknife[seed]

    return st.mean(energy), st.error_jackknife(energy)


def specific_heat(ensemble, temperatures, alpha_tag="AlphasV", beta_tag="BetasV", 
                  eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get the partition function for a given set of temperatures
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """
    Z = moment(ensemble, temperatures, 0, alpha_tag, beta_tag, eigenvalue_tag, modifier)
    E = moment(ensemble, temperatures, 1, alpha_tag, beta_tag, eigenvalue_tag, modifier)
    Q = moment(ensemble, temperatures, 2, alpha_tag, beta_tag, eigenvalue_tag, modifier)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)
    Q_jackknife = st.jackknife(Q)
    
    specific_heat = OrderedDict()
    betas = 1. / temperatures
    for seed in ensemble.seeds:
        specific_heat[seed] = betas**2* ( Q_jackknife[seed] / Z_jackknife[seed] \
                                          - (E_jackknife[seed] / Z_jackknife[seed])**2)

    return st.mean(specific_heat), st.error_jackknife(specific_heat)


def thermodynamics(ensemble, temperatures, alpha_tag="AlphasV", beta_tag="BetasV", 
                   eigenvalue_tag="EigenvaluesV", modifier=None):
    """ Get the partition function for a given set of temperatures
    
    Args:
        ensemble       : Ensemble class
        temperatures   : temperatures as np.array
        tag            : string, which tag is chosen for data
        modifier       : function of quantum number and np.array of eigenvalues
                         how to modify the eigenvalues
        axis           : axis to sum over (default: None, all axes are summed)
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """
    Z = moment(ensemble, temperatures, 0, alpha_tag, beta_tag, eigenvalue_tag, modifier)
    E = moment(ensemble, temperatures, 1, alpha_tag, beta_tag, eigenvalue_tag, modifier)
    Q = moment(ensemble, temperatures, 2, alpha_tag, beta_tag, eigenvalue_tag, modifier)

    Z_jackknife = st.jackknife(Z)
    E_jackknife = st.jackknife(E)
    Q_jackknife = st.jackknife(Q)

    energy = OrderedDict()
    for seed in ensemble.seeds:
        energy[seed] = E_jackknife[seed] / Z_jackknife[seed]
    
    specific_heat = OrderedDict()
    betas = 1. / temperatures
    for seed in ensemble.seeds:
        specific_heat[seed] = betas**2 * ( Q_jackknife[seed] / Z_jackknife[seed] \
                                          - (E_jackknife[seed] / Z_jackknife[seed])**2)

    return st.mean(Z), st.error(Z), st.mean(energy), st.error_jackknife(energy),\
        st.mean(specific_heat), st.error_jackknife(specific_heat)
