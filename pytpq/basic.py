# -*- coding: utf-8 -*-
"""
Basic routines for an ensemble

:author: Alexander Wietek
"""
import numpy as np
from collections import OrderedDict

import pytpq.linalg as pla


def tmatrix(ensemble, seed, qn, alpha_tag="AlphasV", beta_tag="BetasV",
            crop=True, croptol=1e-7, maxdepth=None):
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
    diag = ensemble.data(seed, qn, alpha_tag)[:maxdepth]
    offdiag = ensemble.data(seed, qn, beta_tag)[:maxdepth]
    offdiag = offdiag[:-1]
    # print("qn  ", qn, maxdepth, len(diag), len(offdiag))

    # remove overiterated tmatrix entries
    if crop and np.any(np.abs(offdiag) < croptol):
        diag = diag[:(np.argmax(np.abs(offdiag) < croptol)+1)]
        offdiag = offdiag[:np.argmax(np.abs(offdiag) < croptol)]
    # print("crop", qn, maxdepth, len(diag), len(offdiag))

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
                        shifts=None, ncores=None, maxdepth=None):
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
                                                     alpha_tag=alpha_tag, 
                                                     beta_tag=beta_tag, 
                                                     shifts=shifts,
                                                     maxdepth=maxdepth)
            e0s[seed] = e0
            e0_qns[seed] = e0_qn

    # Parallelization over seeds
    else:
        e0_func = functools.partial(_ground_state_energy_seed,
                                    ensemble=ensemble, alpha_tag=alpha_tag,
                                    beta_tag=beta_tag, shifts=shifts,
                                    maxdepth=maxdepth)
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
                              beta_tag="BetasV", shifts=None,
                              maxdepth=None):
    """ Get the total ground state energy for a seed of an ensemble """
    assert len(ensemble.qns) > 0

    # Prepare data structure holding e0 and its qn
    if shifts == None:
        e0 = np.inf
        e0_qn = ""
    else:
        n_shifts = len(shifts[seed][ensemble.qns[0]])
        e0 = np.full(n_shifts, np.inf)
        e0_qn = [ensemble.qns[0]] * n_shifts

    # find energy minimizing quantum number sector
    for qn in ensemble.qns:
        if ensemble.degeneracy[qn] > 0 and ensemble.dimension[qn] > 0:
            diag, offdiag = tmatrix(ensemble, seed, qn, alpha_tag, 
                                    beta_tag, crop=True,
                                    maxdepth=maxdepth)
            if shifts == None:
                e = pla.tmatrix_e0(diag, offdiag)
                if e < e0:
                    e0 = e
                    e0_qn = qn
            else:
                es = pla.tmatrix_e0(diag, offdiag) + shifts[seed][qn]
                smaller_indices = es < e0
                e0[smaller_indices] = es[smaller_indices]
                for idx, smaller in enumerate(smaller_indices):
                    if smaller:
                        e0_qn[idx] = qn

    return seed, e0, e0_qn
