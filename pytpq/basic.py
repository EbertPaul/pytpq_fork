# -*- coding: utf-8 -*-
"""
Basic routines for an ensemble

:author: Alexander Wietek
"""
import numpy as np
from collections import OrderedDict
import functools
from joblib import Parallel, delayed

import pytpq.linalg as pla


def tmatrix(ensemble, seed, qn, alpha_tag="Alphas", beta_tag="Betas",
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


def ground_state_energy(ensemble, alpha_tag="Alphas", beta_tag="Betas", maxdepth=None, ncores=None):
    """ Get the total ground state energy for all seeds of an ensemble
    
    Args:
        ensemble  : Ensemble class
        alpha_tag : string, which tag is chosen for alpha data
        beta_tag  : string, which tag is chosen for beta data
    Returns:
        OrderedDict, OrderedDict:  dictionaries of ground state energies, 
                                   and corresponding quantum number sector
    """
    e0s = OrderedDict()
    e0_qns = OrderedDict()

    # Get ground state energy for every seed serial
    if ncores == None:
        # Get ground state energy for every seed serial
        for seed in ensemble.seeds:
            _, e0, e0_qn = _ground_state_energy_seed(seed, ensemble, 
                                                        alpha_tag=alpha_tag, 
                                                        beta_tag=beta_tag, 
                                                        maxdepth=maxdepth)
            e0s[seed] = e0
            e0_qns[seed] = e0_qn

    # Parallelization over seeds
    else:
        e0_func = functools.partial(_ground_state_energy_seed,
                                    ensemble=ensemble, alpha_tag=alpha_tag,
                                    beta_tag=beta_tag, maxdepth=maxdepth)

        results = Parallel(n_jobs=ncores, backend="threading")\
                  (map(delayed(e0_func), ensemble.seeds))

        for seed, e0, e0_qn in results:
            e0s[seed] = e0
            e0_qns[seed] = e0_qn

    return e0s, e0_qns



def _ground_state_energy_seed(seed, ensemble, alpha_tag="Alphas", 
                              beta_tag="Betas",
                              maxdepth=None):
    """ Get the total ground state energy for a seed of an ensemble """
    assert len(ensemble.qns) > 0

    # Prepare data structure holding e0 and its qn
    e0 = np.inf
    e0_qn = ""

    # find energy minimizing quantum number sector
    for qn in ensemble.qns:
        if ensemble.degeneracy[qn] > 0 and ensemble.dimension[qn] > 0:
            diag, offdiag = tmatrix(ensemble, seed, qn, alpha_tag, 
                                    beta_tag, crop=True,
                                    maxdepth=maxdepth)
            e = pla.tmatrix_e0(diag, offdiag)
            if e < e0:
                e0 = e
                e0_qn = qn

    return seed, e0, e0_qn
