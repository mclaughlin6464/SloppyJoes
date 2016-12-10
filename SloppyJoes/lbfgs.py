#!/bin/bash
'''Python rewrite of the lbfgs module'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np

def update_storage(k, istep, s,y, d_x, d_jac):

    if np.all(d_x >=1e-15): #else, keep it the same
        if istep <= k:
            s[:, istep] = d_x
            y[:,istep] = d_jac
        else:
            idxs = np.array(range(1, s.shape[1]) )
            s[:, idxs-1] = s[:,idxs] # TODO may need a view here
            y[:, idxs-1] = y[:,idxs]

            s[:, -1] = d_x
            y[:, -1] = d_jac

        return True
    else:
        return False