# -*- coding: utf-8 -*-
"""
Ensemble class

:author: Alexander Wietek
"""

from inspect import isfunction

class Ensemble:
    def __init__(self, tpq_data, qns, degeneracy=None, qn_degeneracy_map=None):

        self.tpq_data = tpq_data
        self.qns = qns
        self.seeds = tpq_data.seeds
        self.degeneracy = dict()
        self.qn_degeneracy_map = dict()
        self.dimension = dict()
        self.exact = dict()

        if degeneracy != None:
            if type(degeneracy) != dict:
                raise ValueError("Need a python dict as degeneracy")
            self.degeneracy = degeneracy
        else:
            for qn in self.qns:
                self.degeneracy[qn] = 1

        if qn_degeneracy_map != None:
            if type(qn_degeneracy_map) != dict:
                raise ValueError("Need a python dict as qn_degeneracy_map")
            self.qn_degeneracy_map = qn_degeneracy_map
        else:
            for qn in self.qns:
                self.qn_degeneracy_map[qn] = qn

        for qn in self.qns:
            self.dimension[qn] = self.tpq_data.dimension(qn_degeneracy_map[qn])
            self.exact[qn]= False
    
    def data(self, seed, qn, tag):
        return self.tpq_data.dataset(seed, self.qn_degeneracy_map[qn])[tag]
       
        
