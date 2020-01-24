# -*- coding: utf-8 -*-
"""
PyTPQ tools for evaluating thermal pure quantum state computations

:author: Alexander Wietek
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import re
from collections import OrderedDict, defaultdict
import h5py
import pylime as pl
            
def get_e0s(data, degeneracies, mu=None, eigenvalue_tag="EigenvaluesV"):
    """ Get the total ground state energy
    
    Args:
        data           : TPQdata as returned by read_data
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict, OrderedDict:  dictionaries of ground state energies, 
                                   and corresponding quantum number sector
    """
    e0s = OrderedDict()
    e0s_qns = OrderedDict()

    # Get ground state energy for every sector
    for seed in data.seeds:
        if mu == None:
            e0s_for_qns = [(data.dataset(seed, qn)[eigenvalue_tag][0], qn) \
                            for qn in data.qns]
        else:
            e0s_for_qns = [(data.dataset(seed, qn)[eigenvalue_tag][0] \
                            - mu * (float(qn[0]) + float(qn[1])), qn) \
                           for qn in data.qns]
        
        e0_qn = min(e0s_for_qns, key=lambda e0_qn: e0_qn[0])    
        e0s[seed] = e0_qn[0]
        e0s_qns[seed] = e0_qn[1]
        
    return e0s, e0s_qns

def get_sum_of_function_on_eigenvalues(data, function, degeneracies, e0s,
                                       eigenvalue_tag="EigenvaluesV"):
    """ Get a sum of a function applied to the eigenvalues
    
    Args:
        data           : TPQdata as returned by read_data
        function       : vectorized function to apply to the eigenvalues
        degeneracies   : function that returns the degeneracy of quantum numbers
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the sums as a funciton of the seed
    """
    # Compute sum for all seeds
    sums = OrderedDict()
    for seed in data.seeds:
        sums[seed] = 0.

        # Accumulate all quantum number sectors
        for qn in data.qns:
            if degeneracies(qn) != 0:
                fun_on_eigs = function(data.dataset(seed, qn)[eigenvalue_tag], 
                                       qn, e0s[seed])
                sums[seed] += np.sum(fun_on_eigs) * degeneracies(qn) * \
                              data.dimension(qn)

    return np.fromiter(sums.values(), dtype=float)


def get_partitions(data, temperature, degeneracies, mu=None, 
                   eigenvalue_tag="EigenvaluesV"):
    """ Get the partition function from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the partition function for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            return np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def get_first_moments(data, temperature, degeneracies, mu=None, 
                      eigenvalue_tag="EigenvaluesV"):
    """ Get the first energy moments from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the first energy moments for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            return eigs * np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return eigs * np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def get_second_moments(data, temperature, degeneracies, mu=None, 
                       eigenvalue_tag="EigenvaluesV"):
    """ Get the second energy moments from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the second enrgy moments for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            return eigs**2 * np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return eigs**2 * np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def get_particle_numbers(data, temperature, degeneracies, mu=None, 
                         eigenvalue_tag="EigenvaluesV"):
    """ Get the particle number from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the particle numbers for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return number * np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return number * np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def get_particle_numbers2(data, temperature, degeneracies, mu=None, 
                          eigenvalue_tag="EigenvaluesV"):
    """ Get the particle number squared from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the particle numbers **2 for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return number**2 * np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            number = float(qns[0]) + float(qns[1])
            return number**2 * np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def get_magnetizations(data, temperature, degeneracies, mu=None, 
                      eigenvalue_tag="EigenvaluesV"):
    """ Get the magnetization from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the magnetization for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            sz = 0.5 * (float(qns[0]) - float(qns[1]))
            return sz * np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            sz = 0.5 * (float(qns[0]) - float(qns[1]))
            number = float(qns[0]) + float(qns[1])
            return sz * np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def get_magnetizations2(data, temperature, degeneracies, mu=None, 
                        eigenvalue_tag="EigenvaluesV"):
    """ Get the magnetization squared from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperature    : temperature at which to evaluate the partition function
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        OrderedDict    : dictionary returning the magnetization squard for seeds
    """
    beta = 1. / temperature
    e0s, _ = get_e0s(data, degeneracies, mu, eigenvalue_tag)
    
    if mu == None:
        def expf(eigs, qns, e0):
            sz = 0.5 * (float(qns[0]) - float(qns[1]))
            return sz**2 * np.exp( -beta * (eigs - e0) )
    else:
        def expf(eigs, qns, e0):
            sz = 0.5 * (float(qns[0]) - float(qns[1]))
            number = float(qns[0]) + float(qns[1])
            return sz**2 * np.exp( -beta * (eigs - mu*number - e0) )
         
    return get_sum_of_function_on_eigenvalues(data, expf, degeneracies,
                                              e0s, eigenvalue_tag)


def partitions(data, temperatures, degeneracies, mu=None, 
               eigenvalue_tag="EigenvaluesV"):
    """ Get the partition function with error estimate from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperatures   : temperatures at which to evaluate partition
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        np.array, np.array: mean and error of partition
    """
    
    partitions_mean = []
    partitions_err = []
    for T in temperatures:
        partitions = get_partitions(data, T, degeneracies, mu=mu, 
                                    eigenvalue_tag=eigenvalue_tag)
        partitions_jck = pl.resample_jackknife(partitions)
        partition_mean_jck = pl.mean(partitions_jck) 
        partition_err_jck = pl.sem_jackknife(partitions_jck)
        partitions_mean.append(partitions_mean_jck)
        partitions_err.append(partitions_err_jck)

    return np.array(partitions_mean), np.array(partitions_err)


def energies(data, temperatures, degeneracies, mu=None, 
             eigenvalue_tag="EigenvaluesV"):
    """ Get the energies with error estimate from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperatures   : temperatures at which to evaluate energy
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        np.array, np.array: mean and error of energy
    """
    
    energies_mean = []
    energies_err = []
    for T in temperatures:
        partitions = get_partitions(data, T, degeneracies, mu=mu, 
                                    eigenvalue_tag=eigenvalue_tag)
        first_moments = get_first_moments(data, T, degeneracies, mu=mu, 
                                          eigenvalue_tag=eigenvalue_tag)

        partitions_jck = pl.resample_jackknife(partitions)
        first_moments_jck = pl.resample_jackknife(first_moments)

        energies_jck = first_moments_jck / partitions_jck
        energy_mean_jck = pl.mean(energies_jck) 
        energy_err_jck = pl.sem_jackknife(energies_jck)
        energies_mean.append(energy_mean_jck)
        energies_err.append(energy_err_jck)

    return np.array(energies_mean), np.array(energies_err)


def specific_heats(data, temperatures, degeneracies, mu=None, 
                   eigenvalue_tag="EigenvaluesV"):
    """ Get the specific heat with error estimate from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperatures   : temperatures at which to evaluate specific heat
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        np.array, np.array: mean and error of specific heat
    """
    
    specific_heats_mean = []
    specific_heats_err = []
    for T in temperatures:
        beta = 1. / T

        partitions = get_partitions(data, T, degeneracies, mu=mu, 
                                    eigenvalue_tag=eigenvalue_tag)
        first_moments = get_first_moments(data, T, degeneracies, mu=mu, 
                                          eigenvalue_tag=eigenvalue_tag)
        second_moments = get_second_moments(data, T, degeneracies, mu=mu, 
                                            eigenvalue_tag=eigenvalue_tag)

        partitions_jck = pl.resample_jackknife(partitions)
        first_moments_jck = pl.resample_jackknife(first_moments)
        second_moments_jck = pl.resample_jackknife(second_moments)

        spec_heat_jck = beta**2 * ( second_moments_jck / partitions_jck - \
                                    (first_moments_jck / partitions_jck)**2 )
        spec_heat_mean_jck = pl.mean(spec_heat_jck) 
        spec_heat_err_jck = pl.sem_jackknife(spec_heat_jck)

        specific_heats_mean.append(spec_heat_mean_jck)
        specific_heats_err.append(spec_heat_err_jck)

    return np.array(specific_heats_mean), np.array(specific_heats_err)


def particle_number(data, temperatures, degeneracies, mu=None, 
                    eigenvalue_tag="EigenvaluesV"):
    """ Get the particle number with error estimate from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperatures   : temperatures at which to evaluate particle number
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        np.array, np.array: mean and error of particle number
    """
    
    numbers_mean = []
    numbers_err = []
    for T in temperatures:
        partitions = get_partitions(data, T, degeneracies, mu=mu, 
                                    eigenvalue_tag=eigenvalue_tag)
        particle_numbers = get_particle_numbers(data, T, degeneracies, mu=mu, 
                                    eigenvalue_tag=eigenvalue_tag)
                
        partitions_jck = pl.resample_jackknife(partitions)
        particle_numbers_jck = pl.resample_jackknife(particle_numbers)

        numbers_jck = particle_numbers_jck / partitions_jck
        numbers_mean_jck = pl.mean(numbers_jck) 
        numbers_err_jck = pl.sem_jackknife(numbers_jck)
        numbers_mean.append(numbers_mean_jck)
        numbers_err.append(numbers_err_jck)

    return np.array(numbers_mean), np.array(numbers_err)


def charge_susceptibility(data, temperatures, degeneracies, mu=None, 
                          eigenvalue_tag="EigenvaluesV"):
    """ Get the charge susceptibility with error estimate from a TPQData set
    
    Args:
        data           : TPQdata as returned by read_data
        temperatures   : temperatures at which to evaluate particle number
        degeneracies   : function that returns the degeneracy of quantum numbers
        mu             : chemical potential, assumed 0 if None
        eigenvalues_tag: string, which tag is chosen for eigenvalues in data
    Returns:
        np.array, np.array: mean and error of particle number
    """
    
    charge_susc_mean = []
    charge_susc_err = []
    for T in temperatures:
        beta = 1. / T

        partitions = get_partitions(data, T, degeneracies, mu=mu, 
                                    eigenvalue_tag=eigenvalue_tag)
        particle_numbers = get_particle_numbers(data, T, degeneracies, mu=mu, 
                                                eigenvalue_tag=eigenvalue_tag)
        particle_numbers2 = get_particle_numbers2(data, T, degeneracies, mu=mu, 
                                                  eigenvalue_tag=eigenvalue_tag)

                
        partitions_jck = pl.resample_jackknife(partitions)
        particle_numbers_jck = pl.resample_jackknife(particle_numbers)
        particle_numbers2_jck = pl.resample_jackknife(particle_numbers2)

        charge_susc_jck = beta * ( particle_numbers2_jck / partitions_jck - \
                                   ( particle_numbers_jck / partitions_jck)**2 )
        charge_susc_mean_jck = pl.mean(charge_susc_jck) 
        charge_susc_err_jck = pl.sem_jackknife(charge_susc_jck)
        charge_susc_mean.append(charge_susc_mean_jck)
        charge_susc_err.append(charge_susc_err_jck)

    return np.array(charge_susc_mean), np.array(charge_susc_err)
