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

    def __init__(self, data, qns_tag="qns", dimension_tag="Dimension"):
        self.data = data
        self.seeds = list(data.keys())

        if len(self.seeds) == 0:
            raise ValueError("No seeds to initialize TPQData")

        # get quantum numbers defined for every seed
        qns_for_each_seed = dict()
        for seed, sectors in self.data.items():
            qns_for_each_seed[seed] = [tuple(map(str, sector)) for sector in sectors]
            
        
        self.qns = qns_for_each_seed[self.seeds[0]]

        # Consistency checks for quantum numbers
        for seed, qns in qns_for_each_seed.items():

            # Check whether quantum numbers are uniquely defined
            qns2 = list(set(qns))
            if len(qns2) != len(qns):
                raise ValueError("Non-unique quantum numbers found for seed", 
                                 seed)

            # Check whether quantum numbers are uniquely defined
            if set(qns) != set(self.qns):
                print(seed)
                print(qns)
                print(self.qns)

                raise ValueError("Not all seeds have the same set"
                                 " of quantum numbers")

        # Check whether dimension is defined for each quantum number sector
        for seed in self.seeds:
            for qn in self.qns:
                if not dimension_tag in self.data[seed][qn].keys():
                    raise ValueError("dimension not defined for seed"
                                     " {} and qn {}".format(seed, qn))

        # Get dimensions from first seed
        self.dimensions = dict()
        for qn in self.qns:
            self.dimensions[qn] = int(self.data[self.seeds[0]][qn]\
                                      [dimension_tag])
                    
        # Check whether dimensions of qns sectors are same across all seeds
        for seed in self.seeds:
            for qn in self.qns:
                if not self.dimensions[qn] == \
                   int(self.data[seed][qn][dimension_tag]):
                    raise ValueError("Qn sector {} does not have same"
                                     " dimension across all seeds".format(qn))
            
      
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
    print("List of files found in all seed directories (first three):", files[:3])
    data_for_seed = OrderedDict()
    if len(files) == 0:
        raise ValueError("No files with \"seed.\" found in directory!")
    
    matched_files = [fl for fl in files if re.search(regex, fl)]
    if verbose:
        print(matched_files)
    print("List of files matching regex (first three):", matched_files[:3])


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
