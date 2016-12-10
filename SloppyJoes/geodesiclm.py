#!/bin/bash
''' Python rewrite of the geodesicLM main module. '''
__author__ = 'Sean McLaughlin'
__email__ = 'swmclau2@stanford.edu'

import numpy as np
from scipy import linalg

from .fdjac import fdjac
from .fdavv import fdavv
from .lambdaFuncs import TrustRegion, Updatedelta_factor, Updatedelta_more, Updatelam_factor, Updatelam_nelson, \
    Updatelam_Umrigar
from .updatejac import updatejac
from .lbfgs import update_storage

#Most of these can be removed, and also absorbed into a kwarg
#i'll make notes of which ones that is
def geodesiclm(func, jacobian, Avv, x, fvec, fjac, n,m,k,callback,info,\
               analytic_jac, analytic_Avv, center_diff, h1, h2, \
                dtd, damp_mode, niters, nfev, njev, jaev, maxiter, maxfev, maxjev,
               maxaev, maxlam, minlam, artol, Cgoal, gtol, xtol, xrtol, ftol, frtol,
               converged, print_level, print_unit, imethod, iaccel, ibold, ibroyden, initialfactor,\
               factoraccept, factorreject, avmax):

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
    alpha = 0.005
    n_accepted = 0

    H_0 = np.eye(n)*0.001
    dtd_inv = np.eye(n)

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
            #Not sure if this only returns lambda
            lam = TrustRegion(fvec, fjac, dtd, delta)

    fvec_new = np.zeros_like(fvec)#i think have to initialize this here
    x_new = np.zeros_like(x) #not sure about these initializations
    Cnew = np.inf

    #main loop
    for istep in xrange(maxiter):

        info = callback(x,v,a,fvec, fjac, acc, lam, dtd, fvec_new, accepted)
        if info != 0:
            converged = -10
            break

        #full or parital jac update
        if accepted>=0 and ibroyden <=0:
            jac_force_update = True
        if accepted+ibroyden <= 0 and not jac_uptodate:
            jac_force_update = True

        if accepted > 0 and ibroyden > 0 and not jac_force_update: #rank deficient
            fjac = updatejac(fjac, fvec, fvec_new, acc, v, a)
            jac_uptodate = False

        if accepted > 0: #accepted step

            #update lbfgs
            fvec = fvec_new

            # TODO this will always be true!
            successful_update = update_storage(k,n_accepted+1, s, y, x_new-x, neg_delta_C_old-neg_delta_C)
            if successful_update: #accepted
                n_accepted+=1

            neg_delta_C_old = neg_delta_C
            x = x_new
            vold = v
            C = Cnew

            if C <= Cbest:
                x_best = x
                Cbest = C
                fvec_best = fvec

        #TODO could have this as a helper function as its repeated from above
        if jac_force_update: #full rank update of the jacobian
            if analytic_jac:
                fjac = jacobian(x)
                njev += 1
            else:
                fjac = fdjac(x, fvec, func, h1, center_diff)
                nfev = nfev + 2 * n if center_diff else nfev + n

        valid_result = np.all(~np.isnan(fjac))

        if valid_result: #no nans, lets party

            jtj = np.dot(fjac, fjac) #not sure if this is the write shape

            #update scaling/lam/trustregion
            if istep > 1:
                if damp_mode == 1:
                    dtd[np.diag_indices(dtd.shape[0])] = np.max(np.vstack(np.diag(jtj), np.diag(dtd)), axis=0)
                #could write a helper function to wrap this up
                if imethod == 0:
                    #update lam directly by fixed factors
                    lam = Updatelam_factor(lam, accepted, factoraccept, factorreject)
                elif imethod == 1:
                    #TODO
                    #update based on gain factor rho (see Nielson reference)
                    lam = Updatelam_nelson(lam, accepted, factoraccept, factorreject, rho)
                elif imethod == 2:
                    #update lam directly using method of Umrigar and Nightingale
                    lam = Updatelam_Umrigar(lam, accepted, v, vold, fvec, fjac, dtd, a_param, Cold, Cnew)
                elif imethod == 10:
                    #update delta by fixed factors
                    delta = Updatedelta_factor(delta, accepted, factoraccept, factorreject)
                    lam = TrustRegion(fvec, fjac, dtd, delta)
                elif imethod == 11:
                    delta, lam = Updatedelta_more(delta, lam, x, dtd, rho, C, Cnew, dirder, actred, av, avmax)
                    lam = TrustRegion(fvec, fjac, dtd, delta)


            #propose step

            g = jtj+lam*dtd
            H_0[np.diag_indices_from(H_0)] = 1.0/np.diag(g)

            dtd_inv[np.diag_indices_from(dtd)] = 1.0/(lam*np.diag(dtd))

            g_upper = linalg.cholesky(g)
            info = 0
        else:#nans in jac
            converged = -11
            break

        if info==0: #decomp successful
            neg_delta_C = -np.dot(fvec, fjac)

            #for now, just get the original version without my additions
            v = linalg.solve((g_upper, False), neg_delta_C)

            temp1 = 0.5*np.dot(v, np.dot(jtj,v))/C
            temp2 = 0.5*lam*np.dot(v, np.dot(dtd,v))/C
            pred_red = temp1+2*temp2
            dirder = -1*(temp1+temp2)
            #calculate cos_alpha, - cos of anlge between step direction and residual
            cos_alpha = np.abs(np.dot(fvec, np.dot(fjac, v)))
            cos_alpha = cos_alpha/np.sqrt(np.dot(fvec, fvec)*np.dot(np.dot(fjac,v), np.dot(fjac, v)))

            if imethod < 10:
                delta = np.sqrt(np.dot(v, np.dot(dtd, v))) #update delta if not set directly

            #update acceleration!

            if iaccel > 0:
                if analytic_Avv:
                    acc = Avv(x,v)
                    naev+=1
                else:
                    acc = FDAvv(x,v,fvec, fjac, func, jac_uptodate, h2)



