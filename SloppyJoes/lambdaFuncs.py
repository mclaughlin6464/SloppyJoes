#!/bin/bash
'''Python rewrite of the lambda module.
Routiens for updating lam'''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

def TrustRegion(fvec, fjac, dtd, delta):
    '''Calls dgqt supplied by minpack to calculate the step and lagarance muliplier'''

    rtol = 1e-3
    atol = 1e-3
    itmax = 10

    jtilde = fjac/np.sqrt(np.diag(dtd))
    gradCtilde = np.dot(fvec, jtilde)
    g = np.dot(jtilde, jtilde)

    fun = lambda x: 0.5*(np.dot(x, np.dot(g, x)))+np.dot(gradCtilde, x)
    constraint = lambda x: delta-norm(x)

    result = minimize(fun,np.zeros_like(gradCtilde), constraints = {'type': 'ineq', 'fun':constraint} )

    x = result.x
    return fun(x)

def Updatelam_factor(lam, accepted, factoraccept, factorreject):
    #update lam based on accepted/rejected step

    return lam/factoraccept if accepted >=0 else lam*factorreject

def Updatelam_nelson(lam, accepted, factoraccept, factorreject, rho):

    if accepted >= 0:
        return lam*max(1/factoraccept, 1-(2*(rho-0.5))**3)
    else:
        return lam*factorreject*(2**(-1*accepted))

def Updatelam_Umrigar(lam, accepted, v, vold, fvec, fjac, dtd, a_param, Cold, Cnew):
    raise NotImplementedError

def Updatedelta_factor(delta, accepted, factoraccept, factorreject):
    #update lam based on accepted/rejected step

    return delta/factoraccept if accepted >=0 else delta*factorreject

def Updatedelta_more(delta, lam, x, dtd, rho, C, Cnew, dirder, actred, av, avmax):

    pnorm = np.sqrt(np.dot(x, np.dot(dtd, x)))
    if rho > 0.25:
        if lam >0 and rho < 0.75:
            temp = 1.0
        else:
            temp = 2.0*pnorm/delta
    else:
        if actred >= 0:
            temp = 0.5
        else:
            temp = 0.5*dirder/(dirder+0.5*actred)
        if 0.01*Cnew >= C or temp < 0.1:
            temp = 0.1

    if av > avmax:
        temp = min(temp, max(avmax/av, 0.1))

    return temp*min(delta, 10*pnorm), lam/temp