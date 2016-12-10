#!/bin/bash
'''Python rewrite of the fdavv module'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np

def fdavv(x,v, fvec, fjac,func, jac_uptodate, h2):

    xtmp = x + h2 * v
    ftmp = func(xtmp)
    if jac_uptodate:
        return (2.0/h2)*( (ftmp-fvec)/h2-np.dot(fjac,v))
    else: #if not up to date, don't use jac
        xtmp = x-h2*v
        acc = func(xtmp)
        return (ftmp-2*fvec+acc)/(h2**2)