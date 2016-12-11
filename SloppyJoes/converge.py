#!/bin/bash
'''Python rewrite of the aconverge module'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

from itertools import izip
import numpy as np

#TODO get the num of arguments down
def convergence_check(converged, accepted, counter, C, Cnew, x, fvec, fjac, lam, xnew,
                      nfev, maxfev, njev, maxjev, naev, maxaev, maxlam, minlam, artol,
                      Cgoal, gtol, xtol, xrtol, ftol,frtol, cos_alpha):

    #the first few should be checked every iteration, since they depend on counts and the
    #Jacobian but not the proposed step

    maxes = [maxfev, maxjev, maxaev, maxlam]
    counts = [nfev, njev, naev, lam]
    conv_vals = [-2, -3, -4, -5]
    count_vals = [0, counter, counter, counter]

    for maxcount, count, conv_val, count_val in izip(maxes, counts, conv_vals, count_vals):

        if maxcount> 0 and count>=maxcount:
            converged = conv_val
            counter = count_val
            return counter, converged

    if minlam > 0 and lam > 0:
        if lam <=minlam:
            counter+=1
            if counter >= 3:
                converged = -6
            return counter, converged

    if artol > 0 and cos_alpha <= artol:
        converged = 1
        return counter, converged

    #if gradient is small
    grad = -1*np.dot(fvec, fjac)
    if np.sqrt(np.sum(grad**2)) <= gtol:
        converged = 3
        return counter, converged

    #if cost is sufficiently small
    if C < Cgoal:
        converged=2
        return counter, converged

    #if step isn't accepted, don't check remaining criteria
    if accepted < 0:
        counter = 0
        converged = 0
        return counter, converged

    #if step size is small
    if np.sqrt(np.dot(x-xnew, x-xnew)) < xtol:
        converged = 4
        return counter, converged

    #if parameter is moving relatively small
    for xx, xxn in izip(x, xnew):
        converged = 5
        if np.abs(xx-xxn) > xrtol*np.abs(xx) or (xxn != xxn):
            converged
            break
    else:
        return counter, converged

    #if cost is not decreasing. Can happen by accidnet, so we require it happen 3 times
    if C-Cnew <= ftol and C-Cnew >=0:
        counter+=1
        if counter>=3:
            converged = 6
        return counter, converged

    #if cost is not decreasing relatively
    if C-Cnew <= frtol*C and C-Cnew >= 0:
        counter+=1
        if counter >= 3:
            converged = 7
        return counter, converged

    counter =0
    converged = 0
    return counter, converged



