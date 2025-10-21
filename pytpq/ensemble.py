# -*- coding: utf-8 -*-
"""
Ensemble class

:author: Alexander Wietek
"""

class Ensemble:
    def __init__(self, tpq_data, qns, degeneracy=None,
                 qn_degeneracy_map=None):

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
            print("No degeneracies were passed to ensemble, assuming 1 for all quantum numbers.")
            for qn in self.qns:
                self.degeneracy[qn] = 1
            

        if qn_degeneracy_map != None:
            if type(qn_degeneracy_map) != dict:
                raise ValueError("Need a python dict as qn_degeneracy_map")
            self.qn_degeneracy_map = qn_degeneracy_map
        else:
            print("No degeneracy map was passed to ensemble, assuming qn -> qn for all quantum numbers.")
            for qn in self.qns:
                self.qn_degeneracy_map[qn] = qn

        for qn in self.qns:
            self.dimension[qn] = self.tpq_data.dimension(qn_degeneracy_map[qn])
            self.exact[qn]= False

        # check if all degeneracies are consistent in the sense that the sum
        # of all degenerate blocks yields the correct Hilbert space dimension
        total_dim = 0
        for qn in self.qns:
            total_dim += self.degeneracy[qn] * self.dimension[qn]
        if self.tpq_data.full_hilbert_space_dim != None:
            if total_dim != self.tpq_data.full_hilbert_space_dim:
                # print degeneracies to the user for debugging
                for qn in self.qns:
                    degeneracy = self.degeneracy[qn]
                    print("qn:", qn, " degeneracy", degeneracy)
                raise ValueError("Inconsistent degeneracies in ensemble! "
                                 "Sum of all degenerate block dimensions"
                                 " is {}, but full Hilbert space dimension"
                                 " given by user is {}".format(
                                total_dim,
                                self.tpq_data.full_hilbert_space_dim))
            print(" ------ HILBERT SPACE DIMENSION CHECKS PASSED ------")
            print("dim H = ", total_dim)
        else:
            print("Dimension of sum of degenerate blocks is {}. " 
                  "Cannot check against user-defined value because 'None' "
                  "was passed to TPQData class. Make sure to check dimension manually!".format(total_dim))

    def data(self, seed, qn, tag):
        return self.tpq_data.dataset(seed, self.qn_degeneracy_map[qn])[tag]
       
        
