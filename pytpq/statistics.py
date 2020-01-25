# -*- coding: utf-8 -*-
"""
Functions to get statistics over random samples

:author: Alexander Wietek
"""
import numpy as np
import scipy as sp
import scipy.stats

def data_dict_to_npdata(data):
    """ convert a dictionary of arrays to a 2d numpy array """
    return np.array([value for key,value in list(data.items())])


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
    return sp.stats.sem(npdata, axis=0)
