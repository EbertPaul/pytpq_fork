import numpy as np
import pytpq.basic as pba
import pytpq.linalg as pla
from collections import OrderedDict

def dos_spectra(ensemble, shifts=None, e0=None, alpha_tag="AlphasV", 
                beta_tag="BetasV", crop=True, croptol=1e-7, maxdepth=None):
    
    if e0 == None:
        e0, _ = pba.ground_state_energy(ensemble, shifts=shifts, alpha_tag=alpha_tag, 
                                        beta_tag=beta_tag)

    pole_list = OrderedDict()
    weight_list = OrderedDict()
    
    for seed in ensemble.seeds:
        
        pole_list[seed] = []
        weight_list[seed] = []

        dim_sum = 0
        for qn in ensemble.qns:
            dim = ensemble.dimension[qn]
            dim_sum += dim
            diag, offdiag = pba.tmatrix(ensemble, seed, qn, alpha_tag, beta_tag, 
                                        crop=crop, croptol=croptol, maxdepth=maxdepth)
            eigs, evecs = pla.tmatrix_eig(diag, offdiag)
            pole_list[seed] += list(eigs - e0[seed])
            # print("s1", np.sum(np.abs(evecs[0,:]**2)))
            assert(np.abs(np.sum(np.abs(evecs[0,:]**2)) - 1.0) < 1e-12)
            weight_list[seed] += list(np.abs(evecs[0,:]**2) * dim)
        
        pole_list[seed] = np.array(pole_list[seed])
        weight_list[seed] = np.array(weight_list[seed]) / dim_sum

        # print("s2", np.sum(weight_list[seed]))
        assert(np.abs(np.sum(weight_list[seed]) - 1.0) < 1e-12)
        assert pole_list[seed].shape[0] == weight_list[seed].shape[0]
    
    return pole_list, weight_list
