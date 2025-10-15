# -*- coding: utf-8 -*-
"""
TPQData class

:author: Alexander Wietek
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import re
from collections import OrderedDict, defaultdict
import h5py
import multiprocessing
import functools
from joblib import Parallel, delayed



class TPQData:

    def __init__(self, data, alpha_tag="Alphas", beta_tag="Betas", dimension_tag="Dimension", eigval_tag="Eigenvalues"):
        self.data = data
        self.seeds = list(data.keys())
        self.alpha_tag = alpha_tag
        self.beta_tag = beta_tag
        self.dimension_tag = dimension_tag
        self.eigval_tag = eigval_tag

        if len(self.seeds) == 0:
            raise ValueError("No seeds to initialize TPQData")

        # get quantum numbers defined for every seed
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
                print(seed)
                print(qns)
                print(self.qns)

                raise ValueError("Not all seeds have the same set"
                                 " of quantum numbers")

        # Check if dimension is defined for all hdf5 files
        for seed in self.seeds:
            for qn in self.qns:
                if not self.dimension_tag in self.data[seed][qn].keys():
                    raise ValueError("dimension not defined for seed"
                                     " {} and qn {}".format(seed, qn))

        # Find smallest dimension of alphas for each quantum number sector
        self.dimensions = dict()
        for qn in self.qns:
            min_dim = np.inf
            for seed in self.seeds:
                new_dim = int(self.data[seed][qn][self.dimension_tag])
                if new_dim < min_dim:
                    min_dim = new_dim
            self.dimensions[qn] = min_dim
       
        # For each quantum number sector, truncate data to smallest dimension across all seeds
        for qn in self.qns:
            for seed in self.seeds:
                print("seed ", seed, " qn ", qn, " dim ", self.data[seed][qn][self.dimension_tag])
                self.data[seed][qn][self.alpha_tag] = self.data[seed][qn][self.alpha_tag][:self.dimensions[qn]]
                self.data[seed][qn][self.beta_tag] = self.data[seed][qn][self.beta_tag][:self.dimensions[qn]]
                self.data[seed][qn][self.eigval_tag] = self.data[seed][qn][self.eigval_tag][:self.dimensions[qn]]
                self.data[seed][qn][self.dimension_tag] = self.dimensions[qn]
            print("Truncated data for quantum number sector", qn, "to dimension", self.dimensions[qn])

                

    def dimension(self, qn):
        """ Return the dimension of a quantum number sector
        
        Args:
            qn         :   quantum number
        Returns:
            int :   dimension of quantum number sector
        """
        return self.dimensions[tuple(map(str, qn))]
      
        
    def dataset(self, seed, qn):
        """ Return the dataset for a given seed and quantum number
        
        Args:
            seed       :   random seed
            qn         :   quantum number
        Returns:
            dictionary :   data for the given seed and quantum number
        """
        return self.data[seed][tuple(map(str, qn))]


def _read_single_file(fl, regex, seed_inds, qn_inds, lime_offset):
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

        # Array data set (remove first dimension for lime)
        else:
            if lime_offset:
                data[key] = hf[key][:][0]
            else:
                data[key] = hf[key][:]
    return seed, tuple(qns), data




def read_data(directory, regex, seed_inds, qn_inds, 
              qns_tag="qns", verbose=True, lime_offset=False,
              ncores=None):
    """ Read data for various seeds and quantities using regular expression
    
    Args:
        directory (str)   : directory containing all data files
        regex (str)       : regular expression to match files in the directory
        seed_inds         : indices in the regex representing the seed index
        qn_inds           : indices in the regex representing the quantum nmbers
        verbose (bool)    : print the matching files
        lime_offset (bool): flag whether data stems from "lime" output
        ncores            : number of parallel cores to read
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
    if ncores == None:
        for fl in matched_files:
            seed, qns, data = _read_single_file(fl, regex, seed_inds, qn_inds, 
                                                lime_offset)

            if seed not in data_for_seed.keys():
                data_for_seed[seed] = OrderedDict()

            data_for_seed[seed][qns] = data
   
    # Parallelization over files
    else:
        read_func = functools.partial(_read_single_file, regex=regex, 
                                      seed_inds=seed_inds, qn_inds=qn_inds, 
                                      lime_offset=lime_offset)

        # with multiprocessing.Pool(ncores) as p:
        #     results = p.map(read_func, matched_files)

        results = Parallel(n_jobs=ncores, backend="threading")\
                  (map(delayed(read_func), matched_files))

        for seed, qns, data in results:
            if seed not in data_for_seed.keys():
                data_for_seed[seed] = OrderedDict()

            data_for_seed[seed][qns] = data                    
                
    return TPQData(data_for_seed)
