#!/bin/bash
'''Python rewrite of the updatejac module.
   Rank-deficient jacobian update'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np

def updatejac(fjac, fvec, fvec_new, acc, v, a):

    r1 = fvec + 0.5*np.dot(fjac, v)+0.125*acc
    djac = 2.0*(r1-fvec-0.5*np.dot(fjac, v))/np.dot(v,v)

    fjac+= np.outer(djac, 0.5*v)

    v2 = 0.5*(v+a)
    djac = 0.5*(fvec_new - r1-np.dot(fjac,v2))/np.dot(v2,v2)

    fjac+= np.outer(djac, v2)

    return fjac