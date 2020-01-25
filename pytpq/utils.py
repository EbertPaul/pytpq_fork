# -*- coding: utf-8 -*-
"""
Utility functions

:author: Alexander Wietek
"""
import numpy as np

def write_fwf(filename, data_dict, width=10, header=None):
    """ Write a fixed width file
    
    Args:
        filename       : filename of text file to write to
        data_dict      : dictionary with data to write
        width          : width of columns in fixed width file
        header         : Additional header
    """
    if header == None:
        header = ""

    savelist = []    
    first = True
    for key, value in data_dict.items():
        if first: 
            header += "{:<23} ".format(key)
            first = False
        else:
            header += "{:<24} ".format(key) 
        savelist.append(value)
    save = np.transpose(np.array(savelist))
    np.savetxt(filename, save, header=header)
            
