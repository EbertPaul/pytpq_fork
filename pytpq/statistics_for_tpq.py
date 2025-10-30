# -*- coding: utf-8 -*-
"""
Functions to get statistics over random samples

:author: Alexander Wietek
"""
import numpy as np
from pytpq import ensemble
import scipy as sp
from collections import OrderedDict
import functools
from joblib import Parallel, delayed

def data_dict_to_npdata(data):
    """ convert a dictionary of arrays to a 2d numpy array """
    return np.array([value for key,value in list(data.items())])


def jackknife(data):
    """ Resample to Jackknife averages """
    npdata = data_dict_to_npdata(data)
    data_resampled = OrderedDict()
    for idx, (seed, array) in enumerate(list(data.items())):
        data_resampled[seed] = \
            np.mean(np.delete(npdata, idx, axis=0), axis=0)

    return data_resampled

def jackknife_parallel(data, ncores=None):
     if ncores == None:
         return jackknife(data)
     else:
          print("Parallelizing jackknife over {} cores.".format(ncores))
          npdata = data_dict_to_npdata(data)

          def jackknife_func(idx, npdata):
               return np.mean(np.delete(npdata, idx, axis=0), axis=0)
          
          parallel_jackknife_func = functools.partial(jackknife_func, npdata=npdata)
          results = Parallel(n_jobs=ncores, backend="threading")\
                    (map(delayed(parallel_jackknife_func), range(len(data))))
          
          data_resampled = OrderedDict()
          for idx, (seed, jackknifed_mean) in enumerate(results):
               data_resampled[seed] = jackknifed_mean
          
          return data_resampled          

def mean(data):
    """ Compute the mean of a data dict of different seeds
    
    Args:
         data :   dict of data for every seed
    Returns:
         np.array: array of means of given data
    """
    npdata = data_dict_to_npdata(data)
    return np.mean(npdata, axis=0)

def error(data):
    """ Compute the error of mean of a data dict of different seeds
    
    Args:
         data :   dict of data for every seed
    Returns:
         np.array: array of error of given data
    """
    npdata = data_dict_to_npdata(data)
    return sp.stats.sem(npdata, axis=0) # computes the standard error of the mean of all jackknife replicates

def error_jackknife(data):
    """ Compute the error of mean of a data dict of different seeds
    
    Args:
         data :   dict of data for every seed
    Returns:
         np.array: array of error of given data
    """
    npdata = data_dict_to_npdata(data)
    return (npdata.shape[0]-1) * sp.stats.sem(npdata, axis=0) # computes the standard error of the mean of all jackknife replicates
