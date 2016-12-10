#!/bin/bash
''' Python rewrite of the geodesicLM main module. '''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

from fdjac import fdjac

#Most of these can be removed, and also absorbed into a kwarg
#i'll make notes of which ones that is
def geodesiclm(func, jacobian, Avv, x, fvec, fjac, n,m,k,callback,info,\
               analytic_jac, analytic_Avv, center_diff, h1, h2, \
                dtd, damp_mode, niters, nfev, njev, jaev, maxiter, maxfev, maxjev,
               maxaev, maxlam, minlam, artol, Cgoal, gtol, xtol, xrtol, ftol, frtol,
               converged, print_level, print_unit, imethod, iaccel, ibold, ibroyden, initialfactor,\
               factoraccept, factorrecject, avmax):

    converged_info = {}

    converged_info[1] = 'artol reached'
    converged_info[2] = 'Cgoal reached'
    converged_info[3] = 'gtol reached'
    converged_info[4] = 'xtol reached'
    converged_info[5] = 'xrtol reached'
    converged_info[6] = 'ftol reached'
    converged_info[7] = 'frtol reached'
    converged_info[-1] = 'maxiters exeeded'
    converged_info[-2] = 'maxfev exceeded'
    converged_info[-3] = 'maxjev exceeded'
    converged_info[-4] = 'maxaev exceeded'
    converged_info[-10] = 'User Termination '
    converged_info[-11] = 'NaN Produced'

    if print_level >=1:
        output_string = '''
        Optimizing with Geodesic-Levenberg-Marquardt algorithm, version 1.1\n
        Method Details:
        Update method: %d
        acceleration: %d
        Bold method: %d
        Broyden updates: %d'''%(imethod, iaccel, ibold, ibroyden)
        print output_string

    niters = 0
    nfev, naev, njaev = 0,0,0
    #think this can be boolean
    converged = 0

    v = np.zeros(n)
    vold = np.zeros_like(v)
    a = np.zeros_like(v)

    cos_alpha = 1.0
    av = 0
    a_param = 0.5

    #initialize our variable storage
    s= np.zeros((n,k))
    y = np.zeros_like(s)
    #not sure if these really need to be initialized
    neg_delta_C, neg_delta_C_old = np.zeros(n), np.zeros(n)

    H_0 = np.eye(n)*0.001

    accepted=0
    counter = 0

    #CALL func(m,n,x,fvec)
    fvec = func(x)

    nfev+=1
    C = 0.5*np.sum(fvec**2) #probably needs to have an axis, depening on shape of fvec

    if print_level >=1:
        print 'Initial Cost: %0.3f'%C

    if np.any(np.isnan(fvec)):
        converged = -11
        maxiter = 0

    Cbest = C
    fvec_best = fvec
    x_best = x

    if analytic_jac:
        fjac = jacobian(x)
        njev+=1
    else:
        fjac = fdjac(x, fvec, func, h1, center_diff)
        nfev = nfev + 2*n if center_diff else nfev + n

    jac_uptodate = True
    jac_force_update = False
    jtj = np.outer(fjac, fjac)

    acc = np.zeros(m) #could move up with the other initializations

    if damp_mode==0:
        dtd = np.eye(n)
    elif damp_mode==1:
        dtd[np.diag_indices(dtd.shape[0])] = np.max(np.vstack(np.diag(jtj), np.diag(dtd)), axis=0)

    if imethod < 10:
        lam = np.max(np.diag(jtj))*initialfactor
    elif imethod >=10:
        delta = initialfactor*np.sqrt(np.dot(x, np.dot(dtd,x)))
        delta = 100.0 if delta == 0.0 else delta
        lam = 1.0

        if converged == 0:
            CALL TrustRegion