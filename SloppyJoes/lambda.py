#!/bin/bash
'''Python rewrite of the lambda module.
Routiens for updating lam'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

def TrustRegion(fvec, fjac, dtd, delta, lam):
    '''Calls dgqt supplied by minpack to calculate the step and lagarance muliplier'''

    rtol = 1e-3
    atol = 1e-3
    itmax = 10

    jtilde = fjac/np.sqrt(np.diag(dtd))