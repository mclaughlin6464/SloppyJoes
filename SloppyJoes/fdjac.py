#!/bin/bash
'''Python rewrite of the fdjac module'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'


def fdjac(x, fvec, func, eps, center_diff):

    epsmach = np.finfo(float).eps
    dx = np.zeros_like(x)
    fjac = []
    if center_diff:
        for i in xrange(x.shape[0]):#TODO vectorize
            h = eps*x[i]
            h = eps if h < epsmach else h
            dx[i] = 0.5
            temp1 = func(x+dx)
            temp2 = func(x-dx)
            fjac.append((temp1-temp2)/h)
            dx[i] = 0.0
    else:
        for i in xrange(x.shape[0]):
            h = eps * x[i]
            h = eps if h < epsmach else h
            dx[i] = h
            temp1 = func(x+dx)
            dx[i] = 0.0
            fjac.append( (temp1-fvec) )/h

    return np.stack(fjac).T #not sure bout the dimension here