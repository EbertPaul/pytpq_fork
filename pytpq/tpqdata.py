# -*- coding: utf-8 -*-
"""
TPQData class

:author: Alexander Wietek
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import re
from collections import OrderedDict
import h5py
import functools



class TPQData:

    def __init__(self, data, full_hilbert_space_dim=None, alpha_tag="Alphas", beta_tag="Betas", dimension_tag="Dimension", eigval_tag="Eigenvalues"):
        self.data = data
        self.seeds = list(data.keys())
        self.alpha_tag = alpha_tag
        self.beta_tag = beta_tag
        self.dimension_tag = dimension_tag
        self.eigval_tag = eigval_tag
        self.full_hilbert_space_dim = full_hilbert_space_dim # allowing consistency check in ensemble class

        if len(self.seeds) == 0:
            raise ValueError("No seeds to initialize TPQData")

        # find quantum numbers for each seed
        qns_for_each_seed = dict()
        for seed, sectors in self.data.items():
            qns_for_each_seed[seed] = [tuple(map(str, sector)) for sector in sectors]
            
        
        self.qns = qns_for_each_seed[self.seeds[0]]

        # Consistency checks for quantum numbers
        for seed, qns in qns_for_each_seed.items():

            # Check whether quantum numbers are unique for each seed
            qns2 = list(set(qns))
            if len(qns2) != len(qns):
                raise ValueError("Non-unique quantum numbers found for seed", 
                                 seed)

            # Check whether quantum numbers are unique across all seeds
            if set(qns) != set(self.qns):
                print("self.qns = ", self.qns, "but seed=", seed, " has qns = ", qns)
                raise ValueError("Not all seeds have the same set"
                                 " of quantum numbers")

        # Check if block dimension is defined for all hdf5 files
        for seed in self.seeds:
            for qn in self.qns:
                if not self.dimension_tag in self.data[seed][qn].keys():
                    raise ValueError("Block dimension not defined for seed"
                                     " {} and qn {}".format(seed, qn))
                
        # check if block dimension is consistent across all seeds for each quantum number sector
        self.dimensions = dict()
        for qn in self.qns:
            dim = int(self.data[self.seeds[0]][qn][self.dimension_tag])
            for seed in self.seeds[1:]:
                new_dim = int(self.data[seed][qn][self.dimension_tag])
                if new_dim != dim:
                    raise ValueError("Inconsistent block dimensions for qn"
                                     " sector {}: dim={} for seed {}, "
                                     "dim={} for seed {}".format(
                                         qn, dim, self.seeds[0],
                                         new_dim, seed))
            self.dimensions[qn] = dim      


                

    def dimension(self, qns):
        """ Return the dimension of a quantum number sector
        
        Args:
            qns         :   quantum number tuple
        Returns:
            int :   dimension of quantum number sector
        """
        return self.dimensions[tuple(map(str, qns))]
      
        
    def dataset(self, seed, qns):
        """ Return the dataset for a given seed and quantum number
        
        Args:
            seed       :   random seed
            qns        :   quantum number tuple
        Returns:
            dictionary :   data for the given seed and quantum number
        """
        return self.data[seed][tuple(map(str, qns))]


def _read_single_file(fl, regex, seed_inds, qn_inds):
    """ reads a single hdf5 file """
    match = re.search(regex, fl)
    group = match.groups()
    qns = []
    for qni in qn_inds:
        qns.append(group[qni])

    seed = group[seed_inds[0]]
    for si in seed_inds:
        assert seed == group[si]

    data = dict()
    hf = h5py.File(fl, 'r')
    for key in hf.keys():
        # Scalar dataset
        if hf[key].shape == tuple():
            data[key] = hf[key][()]
        # vector dataset
        else:
            data[key] = hf[key][:]
    return seed, tuple(qns), data




def read_data(directory, regex, seed_inds, qn_inds, full_hilbert_space_dim=None, verbose=True):
    """ Read data for various seeds and quantities using regular expression
    
    Args:
        directory (str)   : directory containing all data files
        regex (str)       : regular expression to match files in the directory
        seed_inds         : indices in the regex representing the seed index
        qn_inds           : indices in the regex representing the quantum nmbers
        full_hilbert_space_dim (int) : full Hilbert space dimension of the system for consistency checks (e.g. 2^N for N spins-1/2)
        verbose (bool)    : print the matching files
    Returns:
        TPQData:    TPQdata object of quantities for seeds and quantum numbers
    """
    # get files matching the regular expression
    files = []
    for (dirname, _, filenames) in os.walk(directory):
        if "seed." in dirname:
            for fl in filenames:
                files.append(os.path.join(dirname, fl))
    files.sort()
    data_for_seed = OrderedDict()
    if len(files) == 0:
        raise ValueError("No files with \"seed.\" found in directory!")
    
    matched_files = [fl for fl in files if re.search(regex, fl)]
    if verbose:
        print(matched_files)
    print("Number of files matching regex:", len(matched_files))


    # Read files in serial
    for fl in matched_files:
        seed, qns, data = _read_single_file(fl, regex, seed_inds, qn_inds)

        if seed not in data_for_seed.keys():
            data_for_seed[seed] = OrderedDict()

        data_for_seed[seed][qns] = data

    return TPQData(data_for_seed, full_hilbert_space_dim=full_hilbert_space_dim)
